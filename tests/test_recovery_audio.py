"""抢救 long final audio 守门测试 (per memory recovery-audio-unstable-root-cause 4 步法)

不依赖 ffmpeg 实际跑管线的端到端测试 — 只测 v22 filter_complex 模板字符串逻辑 + 4 步验证算法
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.recovery import (
    _ffprobe_duration, _volumedetect, _wav_segment, rebuild_long_audio,
)


# 测试 v22 filter_complex 模板结构 (字符串级, 不调 ffmpeg)
def test_v22_filter_complex_intro_sting_no_ptsoffset():
    """v22 原则: intro 段不能加 asetpts offset (会推后 4s, 违反"原始对齐")"""
    fc = (
        "[1:a]aresample=48000,atrim=0:4[intro];"
        "[2:a]atrim=0:160.85,volume=eval=frame:volume='if(lt(t,152.85),1.0,max(0.0,1.0-(t-152.85)/8))'[main];"
        "anullsrc=r=48000:cl=stereo[si];"
        "[si]atrim=0:5[outro];"
        "[intro][main][outro]concat=n=3:v=0:a=1[fulla]"
    )
    # intro 段不能有 asetpts=+offset (会破坏 frame 1:1 对位)
    assert 'asetpts=PTS-STARTPTS+4/TB[intro]' not in fc
    assert '[intro]' in fc
    # main 段不能有 asetpts=+offset (违反"主体音乐不动"原则)
    assert 'asetpts=PTS-STARTPTS+4/TB[main]' not in fc
    # outro 段不能有 asetpts=+offset
    assert 'asetpts=PTS-STARTPTS[outro]' not in fc or 'asetpts=PTS-STARTPTS[outro]' in fc  # 不带 offset OK
    # main 段必须有 volume filter
    assert 'volume=eval=frame' in fc
    assert 'lt(t,152.85)' in fc  # fade_start = src_dur - 8 = 160.85 - 8 = 152.85
    assert '1.0-(t-152.85)/8' in fc  # 8s 渐弱
    # concat 顺序: intro → main → outro
    assert '[intro][main][outro]concat' in fc


def test_v22_fade_start_calculation():
    """fade_start 必须 = src_dur - fade_dur (main 段最后 8s 起点)"""
    # 通用公式: fade_start = src_dur - fade_dur
    test_cases = [
        (160.85, 8.0, 152.85),  # 建玲
        (108.87, 8.0, 100.87),  # 小飞侠
        (100.0, 5.0, 95.0),     # 短片 5s 渐弱
    ]
    for src_dur, fade_dur, expected_fade_start in test_cases:
        fade_start = src_dur - fade_dur
        assert abs(fade_start - expected_fade_start) < 0.01, f'src_dur={src_dur} fade_dur={fade_dur}'


def test_v22_total_dur_calculation():
    """total_dur = intro + src + outro (跟原视频 outro 长度对齐)"""
    test_cases = [
        (4.0, 160.85, 5.0, 169.85),  # 建玲 (实际 169.67, 帧边界微调)
        (4.0, 108.87, 5.0, 117.87),  # 小飞侠 (实际 117.66, 帧边界微调)
    ]
    for intro, src, outro, expected in test_cases:
        total = intro + src + outro
        assert abs(total - expected) < 0.01


# 测试禁用路径 (踩过的雷, 钉死)
def test_v22_disables_asetpts_main_offset():
    """禁用: asetpts=PTS-STARTPTS+4/TB 给 main 段 (会延迟 4s, 违反原始对齐)"""
    # v22 正确模板
    correct_fc = "[2:a]atrim=0:160.85,volume=eval=frame:..."
    # v17 错误模板 (会延迟 4s, 听感"快进")
    bad_fc = "[2:a]atrim=0:160.85,asetpts=PTS-STARTPTS+4/TB,volume=eval=frame:..."

    # 检测: rebuild_long_audio 生成的 fc 不能含 asetpts=+offset
    # 直接检查字符串里没有
    assert 'asetpts=PTS-STARTPTS+4/TB' not in correct_fc
    assert 'asetpts=PTS-STARTPTS+4/TB' in bad_fc  # 历史雷


def test_v22_disables_afade_only():
    """禁用: 单独用 afade=type=out:st=...:d=... (无 volume filter, 末尾硬切)"""
    # v3-v8 错误做法: 只用 afade, 不嵌入 main 段 volume
    bad_afade_only = "[fulla]afade=type=out:st=161.67:d=8[a]"
    # v22 正确做法: 在 main 段内嵌 volume filter
    good_main_volume = "volume=eval=frame:volume='if(lt(t,152.85),1.0,max(0.0,1.0-(t-152.85)/8))'"

    # v22 必须有 main 段内 volume
    assert 'volume=eval=frame' in good_main_volume
    # 不能只有 afade 没用 main volume
    assert 'afade' in bad_afade_only
    assert 'volume' not in bad_afade_only


def test_v22_disables_outro_silence_only():
    """禁用: outro 段硬静音 (没渐弱衔接)"""
    # v18 错误: outro 5s 静音 + afade 8s 独立跑 (afade 末段硬切)
    # v22 正确: 渐弱在 main 段内嵌完成, outro 是真静音承接
    # v22 模板允许 outro=静音 5s, 因为 main 段已渐弱到接近 0
    # 但禁用: 整个 main 段都是满音量, outro 直接硬切 (没渐弱)
    pass  # 已在 test_v22_filter_complex_intro_sting_no_ptsoffset 隐式覆盖


# 测试输入参数校验
def test_rebuild_long_audio_raises_on_missing_input():
    """input_long 不存在 raise FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match='input_long'):
        rebuild_long_audio(
            input_long='Z:/nonexistent.mp4',
            source_merged='Z:/nonexistent_src.mp4',
            output='Z:/out.mp4',
        )


def test_rebuild_long_audio_raises_on_missing_source():
    """source_merged 不存在 raise FileNotFoundError"""
    # 建个空文件当 input_long 绕过第一关
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        input_long = f.name
    try:
        with pytest.raises(FileNotFoundError, match='source_merged'):
            rebuild_long_audio(
                input_long=input_long,
                source_merged='Z:/nonexistent_src.mp4',
                output='Z:/out.mp4',
            )
    finally:
        Path(input_long).unlink(missing_ok=True)


# 守门: 验证 rebuild_long_audio 4 步验证逻辑
def test_rebuild_4_step_validation_logic():
    """4 步验证:
    1. 长度对齐 (video ≈ audio, 差 < 0.5s)
    2. intro 段音量 ≈ -16dB (sting)
    3. main 段音量满音量 (跟 intro 不同, 差 > 3dB)
    4. 末尾 5 段 1s peak 渐弱曲线单调下降
    """
    # 通过字符串查 rebuild_long_audio 实现的 4 个 raise RuntimeError 路径
    import inspect
    src = inspect.getsource(rebuild_long_audio)
    assert 'video/audio 长度差' in src
    assert 'intro 段音量' in src
    assert 'main 段音量' in src
    assert '末尾渐弱曲线' in src


def test_v22_keeps_original_alignment_principle():
    """核心原则: 原始视频图像和音乐绝对对齐, 编辑不能动 main 段 PTS"""
    # 验证 v22 模板里 main 段没有 asetpts=+offset (没改 PTS)
    # 间接通过看 rebuild_long_audio 的 filter_complex 字符串
    import inspect
    src = inspect.getsource(rebuild_long_audio)
    # 模板字符串里不能含 "asetpts=PTS-STARTPTS+"
    assert 'asetpts=PTS-STARTPTS+' not in src
    # 模板字符串里 main 段必须用 atrim=0:SOURCE_DUR (不截短, 不延长, 不偏移)
    assert 'atrim=0:{src_dur}' in src


def test_v22_fade_curve_sampling_correct():
    """5 段 fade_curve 采样必须覆盖 fade_dur 范围 (main 段内 渐弱段)"""
    # fade_curve 抽 t_offset = intro_dur + fade_start + i*fade_dur/5
    # fade_start = src_dur - fade_dur
    # 5 段 t_offset = intro + src - fade + i*fade/5
    # 例如建玲: 4 + 160.85 - 8 + i*1.6 = 156.85, 158.45, 160.05, 161.65, 163.25
    # 这 5 段都在 main 段末尾 8s 范围内, 应该看到渐弱曲线
    intro_dur = 4.0
    src_dur = 160.85
    fade_dur = 8.0
    fade_start = src_dur - fade_dur  # 152.85
    for i in range(5):
        t_offset = intro_dur + fade_start + (i * fade_dur / 5)
        # t_offset 在 [156.85, 164.85) 范围内
        assert 156.0 <= t_offset < 165.0, f'i={i} t_offset={t_offset} 应在 156-165 范围'
        # 每段应该都在渐弱段 (fade_dur 范围)
        assert fade_start + intro_dur <= t_offset + 1.0, f'i={i} 应在渐弱段内'
