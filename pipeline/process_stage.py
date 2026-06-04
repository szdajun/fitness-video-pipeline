"""进程隔离 Stage 包装器 — 每个 stage 独立进程运行，OS 回收 GPU"""
import importlib, logging, multiprocessing, pickle, sys, traceback


def _reload_disk_caches(ctx):
    """子进程返回后，从磁盘重建 ctx 中的关键数据"""
    import json, cv2, os
    od = str(ctx.output_dir)
    stem = ctx.input_path.stem

    # 1. JSON 缓存文件
    for fname, key in [
        (f"{stem}_keypoints.json", "keypoints"),
        (f"{stem}_cropped_keypoints.json", "cropped_keypoints"),
    ]:
        path = os.path.join(od, fname)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    ctx.set(key, json.load(f))
        except Exception:
            pass

    # 2. 重建 video_info（从源视频重读）
    if not ctx.get("video_info"):
        cap = cv2.VideoCapture(str(ctx.input_path))
        if cap.isOpened():
            ctx.set("video_info", {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            })
            cap.release()


def _is_picklable(v):
    """检查值是否可安全序列化"""
    if v is None: return True
    t = type(v)
    return t in (str, int, float, bool, dict, list, tuple, set)


def _serialize_ctx(ctx) -> bytes:
    """序列化 ctx — 只要路径类数据（走文件系统的大数据跳过）"""
    SKIP = {"keypoints", "cropped_keypoints", "beat_frames",
            "person_count", "motion_data", "sync_data",
            "pose_result", "detections", "skeleton_data"}
    data = {
        "input_path": str(ctx.input_path),
        "output_dir": str(ctx.output_dir),
    }
    for k, v in ctx.data.items():
        if k in SKIP:
            continue
        # 允许小体积结构（dict/list/tuple）传递，只要不太大
        if isinstance(v, (str, int, float, bool, tuple)):
            data[k] = v
        elif isinstance(v, (dict, list)) and len(str(v)) < 10000:
            data[k] = v
    return pickle.dumps(data, protocol=4)


def _deserialize_to_ctx(ctx, data: bytes):
    for k, v in pickle.loads(data).items():
        if k not in ("input_path", "output_dir"):
            ctx.set(k, v)


def _stage_worker(stage_module: str, stage_class: str,
                  ctx_pickle: bytes, config_yaml: str,
                  result_queue: multiprocessing.Queue):
    """子进程：加载 stage → 运行 → pickle 回传结果"""
    try:
        import yaml
        from pipeline.engine import PipelineContext

        ctx_data = pickle.loads(ctx_pickle)

        # 重建 ctx（子进程内独立，不继承父进程 GPU 上下文）
        ctx = PipelineContext(
            input_path=ctx_data.get("input_path", ""),
            output_dir=ctx_data.get("output_dir", "output"),
            config=yaml.safe_load(open(config_yaml, encoding="utf-8")))

        # 把父进程 ctx.data 全部恢复（h2v_size / ken_burns_path / final_path 等）
        # 这是 _serialize_ctx -> pickle -> ctx_data 链路的关键环节
        for k, v in ctx_data.items():
            if k not in ("input_path", "output_dir"):
                ctx.set(k, v)

        # 从磁盘恢复大数据 + video_info（仅在子进程未从 pickle 拿到时补充）
        _reload_disk_caches(ctx)

        # 加载并运行 stage
        mod = importlib.import_module(f"stages.{stage_module}")
        getattr(mod, stage_class)().run(ctx)

        # pickle 回传
        result_queue.put({
            "status": "success",
            "data": _serialize_ctx(ctx),
        })
        sys.exit(0)

    except Exception:
        result_queue.put({
            "status": "error",
            "message": str(sys.exc_info()[1]),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)


class ProcessStage:
    """在独立进程中运行一个 stage — GPU 退出即释放

    支持两种构造：
    - ProcessStage("01_pose_detect", "PoseDetectStage")  # 模块+类名
    - ProcessStage(stage_instance)                        # 从实例提取模块+类名
    """

    def __init__(self, stage_module_or_instance, stage_class: str = None):
        if stage_class is None:
            # 从实例提取
            inst = stage_module_or_instance
            self.stage_module = inst.__class__.__module__.replace("stages.", "")
            self.stage_class = inst.__class__.__name__
            self._name = self.stage_class.replace("Stage", "")
        else:
            self.stage_module = stage_module_or_instance
            self.stage_class = stage_class
            self._name = stage_class.replace("Stage", "")

    def run(self, ctx):
        ctx_pickle = _serialize_ctx(ctx)

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_stage_worker,
            args=(self.stage_module, self.stage_class,
                  ctx_pickle, "config.yaml", result_queue))

        p.start()
        p.join()  # 阻塞等待 → OS 回收子进程 GPU

        if result_queue.empty():
            raise RuntimeError(
                f"Stage {self._name} 进程意外终止 (exit={p.exitcode}), "
                f"可能是 GPU OOM")

        result = result_queue.get()
        if result["status"] == "success":
            _deserialize_to_ctx(ctx, result["data"])
            # 从磁盘回读大数据缓存（keypoints, cropped_keypoints 等）
            _reload_disk_caches(ctx)
            logging.info("[完成] %s (pid=%d)", self._name, p.pid)
        else:
            logging.error("[失败] %s:\n%s", self._name, result["traceback"])
            raise RuntimeError(
                f"Stage {self._name} 失败: {result['message']}")
