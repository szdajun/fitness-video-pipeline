"""教练画像 + SEO 元数据生成

整合所有教练信息（昵称、判词、特点），自动生成 SEO 标题/描述/标签。
"""
import re
import os

# 繁简转换（可选依赖）
try:
    import zhconv

    def to_traditional(text: str) -> str:
        return zhconv.convert(text, "zh-tw")
except ImportError:
    def to_traditional(text: str) -> str:
        return text

DEFAULT_CHANNEL = "细柳营健身"

DEFAULT_SHORTS_POEM = (
    "吹角连营远去\n万家灯火初上\n细柳营中鼎沸\n秦人血脉觉醒\n汗珠子砸地声"
)

DEFAULT_SHORTS_EN = {
    "title": "DAILY AEROBIC WORKOUT",
    "subtitle": "Outdoor Group Fitness",
}

_SHORTS_CTA_EN = [
    "点赞 LIKE & SUBSCRIBE 关注",
    "完整版 Full Workout on Channel",
    "新视频 New Videos Daily",
]

# ── 教练画像 ──────────────────────────────────────────
COACH_PROFILES = {
    "艳青": {
        "nickname": "胭脂虎",
        "judgment": "踏步如虎啸，纤腰扭似涛，酥胸随韵起，刚柔胭脂虎",
        "traits": ["力度强劲", "腰臀线条突出", "柔中带刚"],
        "hook": "暴汗燃脂",
        "workout": "力量燃脂操",
        "focus": "塑腰臀",
        "shorts_focus": "暴汗燃脂",
        "shorts_challenge": "瘦腰瘦腿挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "胭脂虎啸震四方\n踏步如风腰似浪\n刚柔并济铿锵行\n细柳营中第一将",
        "shorts_en_title": "FIERCE CARDIO",
        "shorts_en_subtitle": "FIERCE POWER",
    },
    "丽丽": {
        "nickname": "长安腰女",
        "judgment": "长安腰细若柳摇，腿长随风步步轻，柔姿渐起刚骨架，丽影无双醉银屏",
        "traits": ["腰细腿长", "身姿柔美", "力度渐强"],
        "hook": "长安腰女",
        "workout": "腰腹燃脂操",
        "focus": "打造S曲线",
        "shorts_focus": "暴汗燃脂",
        "shorts_challenge": "瘦腰瘦腿挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "腰若细柳随风摆\n腿如青莲步步开\n柔姿渐起刚骨架\n丽影无双入梦来",
        "shorts_en_title": "WAIST SHREDDER",
        "shorts_en_subtitle": "WAIST SHREDDER",
    },
    "建玲": {
        "nickname": "三宝菩萨",
        "judgment": "时代广场行将令，帅哥美女齐上阵，吉祥三宝福在手，岁月不催韵犹在",
        "traits": ["三孩母亲", "带操利落", "团队领袖", "身材不老"],
        "hook": "高效全身",
        "workout": "全身燃脂操",
        "focus": "产后恢复",
        "shorts_focus": "产后瘦身",
        "shorts_challenge": "宝妈瘦身挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "三宝菩萨气势足\n带操利落不含糊\n岁月不催容颜改\n细柳营中顶梁柱",
        "shorts_en_title": "30s FAT BURN",
        "shorts_en_subtitle": "MOM POWER",
    },
    "小红豆": {
        "nickname": "大唐红线女",
        "judgment": "红豆香汗透罗裳，花枝乱颤舞红妆，娇喘微微惹人怜，酥胸玉臂醉银屏",
        "traits": ["娇小可爱", "女人味足", "动作标准"],
        "hook": "大唐红线女",
        "workout": "全身燃脂操",
        "focus": "居家有氧",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "红豆生来俏模样\n香汗淋漓透红妆\n娇喘微微惹人怜\n花枝乱颤舞霓裳",
        "shorts_en_title": "EASY CARDIO",
        "shorts_en_subtitle": "Beginner Friendly ,  大唐红线女",
    },
    "郭海军": {
        "nickname": "老兵不老",
        "judgment": "老兵卸甲不卸魂，铁骨铮铮踏乐行，岁月不磨豪迈气，操场点兵谁与争",
        "traits": ["退伍军人", "动作刚劲有力", "作风硬朗", "老当益壮"],
        "hook": "老兵不老",
        "workout": "力量燃脂操",
        "focus": "刚劲塑形",
        "shorts_focus": "暴汗燃脂",
        "shorts_challenge": "全身塑形挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "老兵卸甲志不休\n铁骨铮铮弄潮头\n操场点兵威风在\n汗洒细柳写春秋",
        "shorts_en_title": "VETERAN POWER",
        "shorts_en_subtitle": "VETERAN POWER",
    },
    "枫林红": {
        "nickname": "白领丽人",
        "judgment": "丽人一怒百媚生，气场全开霸气横，纤腰玉臂柔中劲，谁人不识枫林红\n鸳鸯袖中藏韬略，胭脂马上请长樱，细柳营中把令行，独领风骚冠群英",
        "traits": ["气场强大", "动作利落干练", "霸气不失女人味"],
        "hook": "白领丽人",
        "workout": "全身燃脂操",
        "focus": "高效有氧",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "丽人迈步气场开\n纤腰玉臂柔中来\n动若脱兔静若松\n枫林红透万千宅",
        "shorts_en_title": "CEO'S FAT BURN",
        "shorts_en_subtitle": "High Energy ,  白领丽人",
    },
    "李刚": {
        "nickname": "托塔天王",
        "judgment": "身如丈八天王，心似低眉菩萨，问君一日所为，晨钟跳到暮鼓",
        "traits": ["魁梧有力", "动作大开大合", "气场沉稳"],
        "hook": "托塔天王",
        "workout": "力量燃脂操",
        "focus": "全身塑形",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "天王托塔镇四方\n铁骨铮铮气宇昂\n步履稳如泰山石\n带操一声万人唱",
        "shorts_en_title": "STRENGTH CARDIO",
        "shorts_en_subtitle": "Full Body Power ,  托塔天王",
    },
    "小飞侠": {
        "nickname": "雷震子",
        "judgment": "雷震双翼踏乐行，节拍入魂韵无穷，举手投足皆律动，风雷一舞万巷空",
        "traits": ["节奏感极强", "动作与音乐完美契合", "律动带动全场"],
        "hook": "跟着音乐",
        "workout": "燃脂操",
        "focus": "律动全身",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "雷震双翼踏乐生\n节拍入魂舞翩跹\n举手投足皆韵律\n风雷一现万巷传",
        "shorts_en_title": "THUNDER BEAT",
        "shorts_en_subtitle": "Full Body Burn ,  雷震子",
        # ====== 自定义三件套文案 (2026-07-13 用户拍板) ======
        "long_description": (
            "【雷震子】小飞侠律动全身操 | 律动全身跟练 | 细柳营健身\n"
            "\n"
            "🔥 跟着节拍跳出汗！雷震子律动全身操，跟着音乐动起来 🔥\n"
            "\n"
            "📌 本期教练：小飞侠（雷震子）\n"
            "📌 训练类型：律动全身\n"
            "📌 节奏卡点，动作流畅，跟着音乐就能跳\n"
            "\n"
            "💪 适合人群：\n"
            "  · 想边听歌边暴汗的打工人\n"
            "  · 节奏感强、爱律动的伙伴\n"
            "  · 想要下班放松、燃脂解压\n"
            "\n"
            "🎵 跟着音乐节拍，节拍入魂舞翩跹\n"
            "   举手投足皆韵律，风雷一现万巷传\n"
            "\n"
            "⏰ 跟练节奏：建议每天 30 分钟\n"
            "   难度：★★★☆☆（律动入门）\n"
            "\n"
            "📍 拍摄地：汉细柳营故地 · 时代广场\n"
            "\n"
            "【胭脂虎健身团】\n"
            "细柳营系列健身操，在历史文化故地\n"
            "用汗水书写当代人的健康生活\n"
            "\n"
            "每晚更新，记得点赞关注！\n"
            "订阅频道：https://youtube.com/@胭脂虎健身团"
        ),
        "douyin_description": (
            "雷震子带操🔥 律动全身燃脂操，跟着音乐暴汗打卡！\n"
            "\n"
            "教练：小飞侠（雷震子）\n"
            "特点：节奏卡点，音乐带动，动作流畅\n"
            "\n"
            "跟着节拍跳就完事了 💪\n"
            "不用想动作，跟着节拍律动起来就行\n"
            "\n"
            "📍 汉细柳营故地 · 时代广场\n"
            "🕐 每晚更新\n"
            "\n"
            "#雷震子 #律动全身 #燃脂操 #细柳营 #胭脂虎健身团"
        ),
        "thumbnail_suggestion": "雷震子·律动",
    },
    "张杰": {
        "nickname": "神行太保",
        "judgment": "万里征途始于足下，飞毛腿疾如风，马拉松魂燃细柳营",
        "traits": ["马拉松跑者", "耐力持久", "节奏稳定"],
        "hook": "神行太保",
        "workout": "燃脂跟练",
        "focus": "持久有氧",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}耐力燃脂 | {channel}",
        "shorts_poem": "天高云淡路远\n帅哥美女争先\n遥见一骑如烟\n细柳营中张哥",
        "shorts_en_title": "ENDURANCE BURN",
        "shorts_en_subtitle": "Marathon Spirit ,  神行太保",
    },
    "彩娥": {
        "nickname": "孤勇者",
        "judgment": "挥袖踏歌领众行，汗沾罗袖亦娉婷。\n一身勇毅承风雨，独护庭前两稚青。",
        "traits": ["勇气可嘉", "动作舒展大方", "坚毅担当", "温暖有力"],
        "hook": "孤勇者",
        "workout": "全身燃脂操",
        "focus": "勇气燃脂",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "挥袖踏歌领众行\n汗沾罗袖亦娉婷\n一身勇毅承风雨\n独护庭前两稚青",
        "shorts_en_title": "FEARLESS CARDIO",
        "shorts_en_subtitle": "Courage & Sweat ,  孤勇者",
    },
    "蜂王": {
        "nickname": "虎痴",
        "judgment": "金顶惹得灯光妒，花臂荡开风雷起，脚下汗水三寸深，方知男儿水做成",
        "traits": ["生猛爆发", "虎气外放", "广场虎将", "节奏如雷"],
        "hook": "生猛爆汗",
        "workout": "生猛操",
        "focus": "生猛爆汗",
        "shorts_focus": "生猛爆汗",
        "shorts_challenge": "生猛挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "金顶夺日光\n花臂扫风雷\n汗雨倾三寸\n虎痴步不回",
        "shorts_en_title": "FEROCIOUS BEAST",
        "shorts_en_subtitle": "Tiger Addict ,  虎痴",
    },
    "李娜": {
        "nickname": "辣妹娜姐",
        "judgment": "华灯初上焰随身，蜜色肌肤透汗津，一跳辣翻半城夏，细柳营里号娜姐",
        "traits": ["火辣活力", "夜场感", "节奏鲜明", "广场辣妹"],
        "hook": "火辣燃脂",
        "workout": "辣妹操",
        "focus": "火辣塑形",
        "shorts_focus": "火辣塑形",
        "shorts_challenge": "火辣挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "华灯初上时\n蜜肌透汗珠\n一跳辣翻夏\n娜姐号细柳",
        "shorts_en_title": "SIZZLE BURN",
        "shorts_en_subtitle": "Hot Lady ,  辣妹娜姐",
    },
    "铁娘子": {
        "nickname": "金刚芭比娃",
        "judgment": "素背凝紫敛腰身，不借脂粉自有神，一跃动时风华起，细柳营中金刚娃",
        "traits": ["素背凝紫", "五分敛腰", "不借脂粉", "风华动人"],
        "hook": "运动风华",
        "workout": "金刚操",
        "focus": "运动风华",
        "shorts_focus": "运动风华",
        "shorts_challenge": "芭比挑战",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "素背凝紫韵\n五分敛腰身\n不借脂粉色\n金刚芭比娃",
        "shorts_en_title": "IRON BARBIE",
        "shorts_en_subtitle": "Iron Barbie ,  金刚芭比娃",
    },
}

# 从 COACH_PROFILES 自动生成昵称映射
_NICKNAME_MAP = {v["nickname"]: k for k, v in COACH_PROFILES.items()}
# 按名称长度降序缓存，用于前缀匹配
_SORTED_COACH_KEYS = sorted(COACH_PROFILES.keys(), key=len, reverse=True)

# ── Focus → 标签 映射 ────────────────────────────────
_FOCUS_TAGS = {
    "塑腰臀": ["瘦腰", "翘臀", "腰腹训练"],
    "打造S曲线": ["S曲线", "瘦腰", "塑形"],
    "产后恢复": ["产后瘦身", "宝妈健身", "腹直肌"],
    "居家有氧": ["居家运动", "有氧操", "在家减肥"],
    "刚劲塑形": ["力量训练", "塑形", "增肌"],
    "高效有氧": ["高效燃脂", "有氧运动", "HIIT"],
    "全身塑形": ["全身塑形", "力量训练", "瘦全身"],
    "律动全身": ["有氧运动", "音乐燃脂", "动感健身"],
    "持久有氧": ["耐力训练", "有氧运动", "跑步辅助"],
    "纤细身段": ["塑形", "优雅有氧", "女性健身"],
}


def _clean_input_name(name: str) -> str:
    """去除名称末尾的数字、下划线、短横线、点、空格及后续所有字符。

    匹配数字/下划线/短横线/点/空格中的任意一个，从匹配点截断后续所有字符。
    例: "小红豆4.mp4" → "小红豆"  （数字4被截断，.mp4因splitext已去除）
    """
    return re.sub(r'[\d_\-.\s].*$', '', str(name))


def _match_coach_key(stem: str):
    """从 stem 匹配最长的教练 key，未匹配返回 None"""
    for key in _SORTED_COACH_KEYS:
        if key.startswith(stem) or stem.startswith(key) or stem in key or key in stem:
            return key
    return None


def _resolve_coach_name(name: str) -> str:
    """统一识别: 文件名或简称 → 完整教练名 (如 '海军3' → '郭海军', '海军_danmaku' → '郭海军')

    2026-06-17 修: 之前 _clean_input_name 把"海军3_danmaku_burst"截成"海军",
    _match_coach_key("海军") 又匹配不到 "郭海军" (因为 in/startswith 方向错),
    导致 get_coach() 返回默认画像 (无专属诗词/英文名).

    解决: 先用 detect_coach_from_filename 走完整匹配逻辑, 失败再用 _match_coach_key + 简单包含测试
    """
    # 1. 用完整 detect 流程 (含文件名清理)
    detected = detect_coach_from_filename(name)
    if detected and detected in COACH_PROFILES:
        return detected
    # 2. 退化: 直接在 stem 上找最长 key 包含
    stem = _clean_input_name(name)
    # 2026-06-29 BUGFIX: stem 为空 (空名/纯数字文件名) 时, `key.startswith("")` 和
    # `"" in key` 恒真 → 误返回字典首个最长 key (小红豆), 导致 get_coach("") 串词
    # (ShortsStage 未传 --shorts-coach 时片头诗词串成小红豆). 空 stem 不匹配任何教练.
    if not stem:
        return stem
    if stem in COACH_PROFILES:
        return stem
    if key := _match_coach_key(stem):
        return key
    # 3. 找最长 key 在 stem 任意位置出现
    for key in _SORTED_COACH_KEYS:
        if key in stem or stem in key:
            return key
    return stem


def get_coach(name: str) -> dict:
    """从教练名或昵称或文件名中查找教练画像。

    匹配顺序: 精确名称 → 前缀匹配（长优先）→ 昵称反向查找 → 默认画像。
    默认画像包含 name 本身，其他字段为通用值，确保调用方不会因缺键报错。

    Args:
        name: 教练名（如"艳青"）、昵称（如"胭脂虎"）或文件名（如"艳青4.mp4"）。

    Returns:
        教练画像 dict，含 name/nickname/judgment/traits/hook/workout/focus/title_tpl。
    """
    resolved = _resolve_coach_name(name)
    if resolved in COACH_PROFILES:
        return {"name": resolved, **COACH_PROFILES[resolved]}

    if resolved in _NICKNAME_MAP:
        real_name = _NICKNAME_MAP[resolved]
        return {"name": real_name, **COACH_PROFILES[real_name]}

    return {
        "name": resolved, "nickname": resolved, "judgment": "",
        "traits": [], "hook": "全身燃脂", "workout": "燃脂操",
        "focus": "暴汗燃脂",
        "title_tpl": "【{nickname}】{name}{workout} | {focus}跟练 | {channel}",
    }


def detect_coach_from_filename(filename: str) -> str:
    """从文件名中提取教练名

    Args:
        filename: 视频文件名（如"小红豆4.mp4"）

    Returns:
        教练名（如"小红豆"），未识别时返回文件名去数字部分
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    stem = _clean_input_name(stem)
    if not stem:
        return ""  # 纯数字/符号开头文件名 (如 2026-06-22_xxx) 截断后为空 → 返回空让下游 skip,
                   # 而非被 _match_coach_key("") 误匹配成某教练 (会导致给无关视频误换脸)
    key = _match_coach_key(stem)
    return key if key else stem


def get_shorts_poem(name: str) -> str:
    """获取教练短视频叠加诗词。

    用于 Shorts 视频中显示的教练专属大字诗词。未匹配到教练时返回默认诗词。

    Args:
        name: 教练名或文件名。

    Returns:
        诗词字符串（含换行符），如 "胭脂虎啸震四方\n踏步如风腰似浪\n..."
    """
    coach = get_coach(name)
    return coach.get("shorts_poem", DEFAULT_SHORTS_POEM)


def get_shorts_en(name: str) -> dict:
    """获取教练 Shorts 英文标题和副标题。

    Args:
        name: 教练名或文件名。

    Returns:
        {"title": "30s FAT BURN 🔥", "subtitle": "3-Mommy Coach ,  三宝妈"}
    """
    coach = get_coach(name)
    title = coach.get("shorts_en_title", DEFAULT_SHORTS_EN["title"])
    subtitle = coach.get("shorts_en_subtitle", DEFAULT_SHORTS_EN["subtitle"])
    return {"title": title, "subtitle": subtitle}


def get_shorts_cta_en() -> list:
    """获取 Shorts 结尾双语 CTA 文字列表。"""
    return list(_SHORTS_CTA_EN)


def generate_title(coach: dict, config: dict) -> str:
    """根据教练画像和配置生成 SEO 标题。

    格式: 【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}

    Args:
        coach: get_coach() 返回的教练画像 dict。
        config: SEO 配置 dict（需含 channel 字段，可选）。

    Returns:
        标题字符串，如 "【暴汗燃脂】胭脂虎艳青力量燃脂操 | 塑腰臀跟练 | 细柳营健身"
    """
    channel = config.get("channel", DEFAULT_CHANNEL)
    base_title = coach["title_tpl"].format(
        name=coach["name"],
        nickname=coach["nickname"],
        channel=channel,
        hook=coach.get("hook", "暴汗燃脂"),
        workout=coach.get("workout", "燃脂操"),
        focus=coach.get("focus", "全身燃脂"),
    )
    # 追加搜索关键词（户外/零基础/不伤膝/跟练）
    keywords = config.get("seo_keywords", "户外燃脂操｜零基础不伤膝｜跟练打卡")
    title = f"{base_title}｜{keywords}"
    # 繁体转换
    if config.get("traditional", False):
        title = to_traditional(title)
    return title


def generate_douyin_title(name: str) -> str:
    """生成抖音短标题，限 30 字。格式：【花名】简短描述"""
    coach = get_coach(name)
    nickname = coach.get("nickname", name)
    workout = coach.get("workout", "燃脂操")
    focus = coach.get("focus", "全身燃脂")
    return f"【{nickname}】{focus}{workout}"[:30]


def generate_douyin_description(name: str) -> str:
    """生成抖音简介文案，含判词 + CTA + 话题"""
    coach = get_coach(name)
    nickname = coach.get("nickname", name)
    poem_lines = coach.get("shorts_poem", "").split("\n")
    poem_short = "，".join(poem_lines[:2]) if poem_lines else ""
    return (
        f"{nickname}{coach['name']}带操！{poem_short}。"
        "汉细柳营故地, 时代广场，每天跟练暴汗燃脂，零基础也能跳！"
        f"#胭脂虎健身团 #{nickname} #燃脂操 #跟练 #健身"
    )


def generate_description(coach: dict, config: dict, duration: str = "") -> str:
    """根据教练画像和配置生成 SEO 描述。

    结构: 判词 → 教练介绍 → 本期亮点(风格/强度/适合) → 时长 → 时间轴 → CTA → 话题标签

    优先用 coach['long_description'] (教练自定义); 没设时走通用模板.

    Args:
        coach: get_coach() 返回的教练画像 dict。
        config: SEO 配置 dict（需含 channel/intensity/audience/tags 字段，可选）。
        duration: 视频时长字符串（如 "30分钟"），为空则不显示。

    Returns:
        多行描述字符串。
    """
    # 优先用教练自定义 long_description (2026-07-13 雷震子首用)
    if coach.get("long_description"):
        return coach["long_description"]

    channel = config.get("channel", DEFAULT_CHANNEL)
    intensity = config.get("intensity", "中等强度")
    audience = config.get("audience", "所有水平, 新手友好")
    tags = config.get("tags", [])

    lines = []
    if coach["judgment"]:
        lines.append(coach["judgment"])
        lines.append("")

    lines.append(f"{coach['nickname']}{coach['name']} 带你暴汗燃脂！")
    lines.append("")
    lines.append("本期亮点：")
    lines.append(f"  - {coach['nickname']}{coach['name']}教练({coach['hook']})领操")
    lines.append(f"  - 风格：{', '.join(coach['traits'][:3])}")
    lines.append(f"  - 强度：{intensity}")
    lines.append(f"  - 适合：{audience}")
    lines.append("")
    if duration:
        lines.append(f"时长：{duration}")
        lines.append("")
    lines.append("视频时间轴：")
    lines.append("00:00 暖身准备")
    lines.append("xx:xx 主运动开始（视实际视频更新）")
    lines.append("xx:xx 拉伸放松（视实际视频更新）")
    lines.append("")
    lines.append(f"关注{channel}，每天带你练！新视频每周更新。")
    lines.append("")
    if tags:
        lines.append(" ".join(f"#{t}" for t in tags))

    desc = "\n".join(lines)
    if config.get("traditional", False):
        desc = to_traditional(desc)
    return desc


def generate_tags(coach: dict, config: dict) -> list:
    """根据教练画像和配置生成 SEO 标签列表。

    组合来源: 频道通用标签 + Focus 身体部位标签 + 教练特点标签 + 教练名 + 昵称。
    自动去重，不保证顺序。

    Args:
        coach: get_coach() 返回的教练画像 dict。
        config: SEO 配置 dict（需含 tags 字段）。

    Returns:
        去重后的标签列表。
    """
    tags = set(config.get("tags", []))

    focus = coach.get("focus", "")
    if focus in _FOCUS_TAGS:
        tags.update(_FOCUS_TAGS[focus])

    for t in coach["traits"]:
        tag = t.strip()
        if tag:
            tags.add(tag)

    if coach["name"]:
        tags.add(coach["name"])
    if coach["nickname"]:
        tags.add(coach["nickname"])

    return list(tags)
