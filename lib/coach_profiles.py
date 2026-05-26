"""教练画像 + SEO 元数据生成

整合所有教练信息（昵称、判词、特点），自动生成 SEO 标题/描述/标签。
"""
import re
import os

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
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "胭脂虎啸震四方\n踏步如风腰似浪\n刚柔并济铿锵行\n细柳营中第一将",
        "shorts_en_title": "FIERCE CARDIO 🔥",
        "shorts_en_subtitle": "Power & Grace · 胭脂虎",
    },
    "丽丽": {
        "nickname": "腰女",
        "judgment": "腰细若柳摇金殿，腿长随风步步轻，柔姿渐起刚骨架，丽影无双醉银屏",
        "traits": ["腰细腿长", "身姿柔美", "力度渐强"],
        "hook": "极致瘦腰",
        "workout": "腰腹燃脂操",
        "focus": "打造S曲线",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "腰若细柳随风摆\n腿如青莲步步开\n柔姿渐起刚骨架\n丽影无双入梦来",
        "shorts_en_title": "WAIST SHREDDER 🔥",
        "shorts_en_subtitle": "S-Curve Sculpt · 腰女",
    },
    "建玲": {
        "nickname": "三宝妈",
        "judgment": "时代广场行将令，帅哥美女齐上阵，吉祥三宝福在手，岁月不催韵犹在",
        "traits": ["三孩母亲", "带操利落", "团队领袖", "身材不老"],
        "hook": "高效全身",
        "workout": "全身燃脂操",
        "focus": "产后恢复",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "三宝妈来气势足\n带操利落不含糊\n岁月不催容颜改\n细柳营中顶梁柱",
        "shorts_en_title": "30s FAT BURN 🔥",
        "shorts_en_subtitle": "3-Mommy Coach · 三宝妈",
    },
    "小红豆": {
        "nickname": "红娘子",
        "judgment": "红豆香汗透罗裳，花枝乱颤舞红妆，娇喘微微惹人怜，酥胸玉臂醉银屏",
        "traits": ["娇小可爱", "女人味足", "动作标准"],
        "hook": "新手友好",
        "workout": "全身燃脂操",
        "focus": "居家有氧",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "红豆生来俏模样\n香汗淋漓透红妆\n娇喘微微惹人怜\n花枝乱颤舞霓裳",
        "shorts_en_title": "EASY CARDIO 🌸",
        "shorts_en_subtitle": "Beginner Friendly · 红娘子",
    },
    "郭海军": {
        "nickname": "老兵不老",
        "judgment": "老兵卸甲不卸魂，铁骨铮铮踏乐行，岁月不磨豪迈气，操场点兵谁与争",
        "traits": ["退伍军人", "动作刚劲有力", "作风硬朗", "老当益壮"],
        "hook": "老兵不老",
        "workout": "力量燃脂操",
        "focus": "刚劲塑形",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "老兵卸甲志不休\n铁骨铮铮弄潮头\n操场点兵威风在\n汗洒细柳写春秋",
        "shorts_en_title": "VETERAN POWER 💪",
        "shorts_en_subtitle": "Never Too Old · 老兵不老",
    },
    "枫林红": {
        "nickname": "霸道总裁",
        "judgment": "总裁一怒百媚生，气场全开霸气横，纤腰玉臂柔中劲，谁人不识枫林红\n鸳鸯袖中藏韬略，胭脂马上请长樱，细柳营中把令行，帅哥美女齐上阵",
        "traits": ["气场强大", "动作利落干练", "霸气不失女人味"],
        "hook": "霸道总裁",
        "workout": "全身燃脂操",
        "focus": "高效有氧",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "总裁迈步气场开\n纤腰玉臂柔中来\n动若脱兔静若松\n枫林红透万千宅",
        "shorts_en_title": "CEO'S FAT BURN 🔥",
        "shorts_en_subtitle": "High Energy · 霸道总裁",
    },
    "李刚": {
        "nickname": "托塔天王",
        "judgment": "身如丈八天王，心似低眉菩萨，问君一日所为，晨钟跳到暮鼓",
        "traits": ["魁梧有力", "动作大开大合", "气场沉稳"],
        "hook": "托塔天王",
        "workout": "力量燃脂操",
        "focus": "全身塑形",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "天王托塔镇四方\n铁骨铮铮气宇昂\n步履稳如泰山石\n带操一声万人唱",
        "shorts_en_title": "STRENGTH CARDIO 💪",
        "shorts_en_subtitle": "Full Body Power · 托塔天王",
    },
    "小飞侠": {
        "nickname": "节拍战神",
        "judgment": "飞侠踏乐步生风，节拍入魂韵无穷，举手投足皆律动，战神一舞万巷空",
        "traits": ["节奏感极强", "动作与音乐完美契合", "律动带动全场"],
        "hook": "跟着音乐",
        "workout": "燃脂操",
        "focus": "律动全身",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "飞侠踏乐步生烟\n节拍入魂舞翩跹\n举手投足皆韵律\n战神一现万巷传",
        "shorts_en_title": "BEAT SYNC CARDIO 💥",
        "shorts_en_subtitle": "Full Body Burn · 节拍战神",
    },
    "张杰": {
        "nickname": "飞毛腿",
        "judgment": "万里征途始于足下，飞毛腿疾如风，马拉松魂燃细柳营",
        "traits": ["马拉松跑者", "耐力持久", "节奏稳定"],
        "hook": "飞毛腿",
        "workout": "燃脂跟练",
        "focus": "持久有氧",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}耐力燃脂 | {channel}",
        "shorts_poem": "天高云淡路远\n帅哥美女争先\n遥见一骑如烟\n细柳营中张哥",
        "shorts_en_title": "ENDURANCE BURN 🏃",
        "shorts_en_subtitle": "Marathon Spirit · 飞毛腿",
    },
    "艳玲": {
        "nickname": "俏玲珑",
        "judgment": "玲珑身段柔中刚，娇俏带操步步香，一笑倾城细柳营\n恍惚间似花枝乱颤，耳畔闻娇喘微微，眼见她香汗淋漓，面如桃花肤似雪",
        "traits": ["身段玲珑", "娇俏带操", "柔中带刚"],
        "hook": "全身塑形",
        "workout": "塑形燃脂操",
        "focus": "纤细身段",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}优雅跟练 | {channel}",
        "shorts_poem": "玲珑身段柔中刚\n娇俏带操步步香\n细柳营中花一朵\n一笑倾城压群芳",
        "shorts_en_title": "GRACEFUL BURN ✨",
        "shorts_en_subtitle": "Elegant Sculpt · 俏玲珑",
    },
    "彩娥": {
        "nickname": "孤勇者",
        "judgment": "挥袖踏歌领众行，汗沾罗袖亦娉婷。\n一身勇毅承风雨，独护庭前两稚青。",
        "traits": ["勇气可嘉", "动作舒展大方", "坚毅担当", "温暖有力"],
        "hook": "孤勇者",
        "workout": "全身燃脂操",
        "focus": "勇气燃脂",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
        "shorts_poem": "挥袖踏歌领众行\n汗沾罗袖亦娉婷\n一身勇毅承风雨\n独护庭前两稚青",
        "shorts_en_title": "FEARLESS CARDIO ⚡",
        "shorts_en_subtitle": "Courage & Sweat · 孤勇者",
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
        if key.startswith(stem) or stem.startswith(key):
            return key
    return None


def get_coach(name: str) -> dict:
    """从教练名或昵称或文件名中查找教练画像。

    匹配顺序: 精确名称 → 前缀匹配（长优先）→ 昵称反向查找 → 默认画像。
    默认画像包含 name 本身，其他字段为通用值，确保调用方不会因缺键报错。

    Args:
        name: 教练名（如"艳青"）、昵称（如"胭脂虎"）或文件名（如"艳青4.mp4"）。

    Returns:
        教练画像 dict，含 name/nickname/judgment/traits/hook/workout/focus/title_tpl。
    """
    stem = _clean_input_name(name)
    if stem in COACH_PROFILES:
        return {"name": stem, **COACH_PROFILES[stem]}

    key = _match_coach_key(stem)
    if key:
        return {"name": key, **COACH_PROFILES[key]}

    if stem in _NICKNAME_MAP:
        real_name = _NICKNAME_MAP[stem]
        return {"name": real_name, **COACH_PROFILES[real_name]}

    return {
        "name": stem, "nickname": stem, "judgment": "",
        "traits": [], "hook": "全身燃脂", "workout": "燃脂操",
        "focus": "暴汗燃脂",
        "title_tpl": "【{hook}】{nickname}{name}{workout} | {focus}跟练 | {channel}",
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
        {"title": "30s FAT BURN 🔥", "subtitle": "3-Mommy Coach · 三宝妈"}
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
    return coach["title_tpl"].format(
        name=coach["name"],
        nickname=coach["nickname"],
        channel=channel,
        hook=coach.get("hook", "暴汗燃脂"),
        workout=coach.get("workout", "燃脂操"),
        focus=coach.get("focus", "全身燃脂"),
    )


def generate_description(coach: dict, config: dict, duration: str = "") -> str:
    """根据教练画像和配置生成 SEO 描述。

    结构: 判词 → 教练介绍 → 本期亮点(风格/强度/适合) → 时长 → 时间轴 → CTA → 话题标签

    Args:
        coach: get_coach() 返回的教练画像 dict。
        config: SEO 配置 dict（需含 channel/intensity/audience/tags 字段，可选）。
        duration: 视频时长字符串（如 "30分钟"），为空则不显示。

    Returns:
        多行描述字符串。
    """
    channel = config.get("channel", DEFAULT_CHANNEL)
    intensity = config.get("intensity", "中等强度")
    audience = config.get("audience", "所有水平 / 新手友好")
    tags = config.get("tags", [])

    lines = []
    if coach["judgment"]:
        lines.append(coach["judgment"])
        lines.append("")

    lines.append(f"{coach['nickname']}{coach['name']} 带你暴汗燃脂！")
    lines.append("")
    lines.append("本期亮点：")
    lines.append(f"  - {coach['nickname']}{coach['name']}教练({coach['hook']})领操")
    lines.append(f"  - 风格：{'·'.join(coach['traits'][:3])}")
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

    return "\n".join(lines)


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
