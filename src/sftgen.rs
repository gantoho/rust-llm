//! 合成 SFT 对话语料的生成器（纯 Rust 实现，等价于原 `tools/gen_sft_corpus.py`）。
//!
//! **为什么需要它**：仓库自带的对话语料只有约 1.4 万 token，切 10% 做验证集后每轮评估只覆盖
//! 1.4k token，早停信号噪声极大。这里用"领域 × 槽位"文法生成结构多样的中文短对话，
//! 把语料扩到几十万 token，让 SFT 的验证曲线真正可用。
//!
//! **诚实声明**：这些是**合成语料**，只能教格式、领域词汇与基本话术，不能替代真实人工对话。
//! 产物写到 `data/sft/zh_synth_dialogue.txt`（`train.sft_file` 里的 `data/sft/` 目录会自动收录）。
//!
//! 用法（由 `sft-gen` 子命令调用）：
//!
//! ```text
//! cargo run --release -- sft-gen --dialogues 4000 --seed 42
//! ```

use crate::rng::Rng;

const USER: &str = "用户：";
const ASSISTANT: &str = "助手：";

// ---- 槽位词表 -------------------------------------------------------------
const SURNAMES: &[&str] = &["张", "李", "王", "刘", "陈", "赵", "周", "吴"];
const DAYS: &[&str] = &[
    "今天",
    "明天",
    "后天",
    "本周五",
    "本周六",
    "下周一",
    "下周三",
];
const TIMES: &[&str] = &[
    "上午九点",
    "上午十点半",
    "中午十二点",
    "下午两点",
    "下午四点",
    "晚上七点",
    "晚上八点半",
];
const PEOPLE: &[&str] = &["一位", "两位", "三位", "四位", "五位", "六位"];
const AMOUNTS: &[&str] = &[
    "一百元",
    "两百元",
    "五百元",
    "一千元",
    "两千元",
    "五千元",
    "一万元",
];
const CARDS: &[&str] = &["储蓄卡", "信用卡", "工资卡", "借记卡"];
const DISHES: &[&str] = &[
    "宫保鸡丁",
    "麻婆豆腐",
    "红烧肉",
    "清蒸鲈鱼",
    "番茄牛腩",
    "酸辣土豆丝",
    "水煮鱼",
    "蒜蓉西兰花",
];
const DRINKS: &[&str] = &[
    "一杯红茶",
    "一杯柠檬水",
    "一杯豆浆",
    "一壶普洱",
    "一瓶气泡水",
];
const CITIES: &[&str] = &[
    "北京", "上海", "杭州", "成都", "广州", "西安", "厦门", "青岛",
];
const TRANSPORT: &[&str] = &["高铁", "飞机", "动车", "大巴"];
const SYMPTOMS: &[&str] = &["头疼", "嗓子疼", "咳嗽", "低烧", "胃疼", "失眠", "腰疼"];
const DEPARTMENTS: &[&str] = &["内科", "外科", "耳鼻喉科", "骨科", "消化科"];
const PRODUCTS: &[&str] = &[
    "蓝牙耳机",
    "机械键盘",
    "保温杯",
    "电饭煲",
    "移动电源",
    "鼠标垫",
];
const ISSUES: &[&str] = &[
    "连不上网",
    "开机很慢",
    "屏幕闪烁",
    "声音断断续续",
    "电池掉得很快",
];
const TOPICS: &[&str] = &[
    "人工智能",
    "天气",
    "读书",
    "运动",
    "旅行",
    "编程",
    "做饭",
    "电影",
];

fn pick<'a>(rng: &mut Rng, xs: &[&'a str]) -> &'a str {
    xs[rng.choice(xs.len())]
}

/// 一轮问答
type Turn = (String, String);

fn restaurant(rng: &mut Rng) -> Vec<Turn> {
    let n = pick(rng, PEOPLE);
    let (day, tm) = (pick(rng, DAYS), pick(rng, TIMES));
    let dish = pick(rng, DISHES);
    let drink = pick(rng, DRINKS);
    let surname = pick(rng, SURNAMES);
    let name = format!("{surname}先生");
    vec![
        (
            "你好，我想订个位子。".into(),
            "您好！请问几位，什么时间到？".into(),
        ),
        (
            format!("{n}，{day}{tm}。"),
            format!("好的，{day}{tm}，{n}，我记下了。请问怎么称呼您？"),
        ),
        (
            format!("姓{surname}。"),
            format!("{name}您好，位子已经留好了，到店报名字就行。"),
        ),
        (
            "你们有什么推荐的菜吗？".into(),
            format!("我们的{dish}很受欢迎，另外{dish}的份量也够两个人。"),
        ),
        (
            format!("那就来一份{dish}，再{drink}。"),
            format!("好的，一份{dish}，{drink}，马上安排。"),
        ),
        ("谢谢。".into(), format!("不客气，期待您{day}光临。")),
    ]
}

fn bank(rng: &mut Rng) -> Vec<Turn> {
    let (card, amt) = (pick(rng, CARDS), pick(rng, AMOUNTS));
    vec![
        (
            "你好，我想查一下余额。".into(),
            "好的，请出示您的银行卡或者身份证。".into(),
        ),
        (
            format!("我用{card}。"),
            "好的，请插卡并输入密码。您想查余额还是取款？".into(),
        ),
        (
            format!("我想取{amt}。"),
            format!("好的，取{amt}。请问需要打印凭条吗？"),
        ),
        (
            "要一张。".into(),
            "好的，请稍等。请确认金额，然后取走现金和凭条。".into(),
        ),
        ("取好了，谢谢。".into(), "不客气，欢迎下次光临。".into()),
    ]
}

fn travel(rng: &mut Rng) -> Vec<Turn> {
    let (city, transport) = (pick(rng, CITIES), pick(rng, TRANSPORT));
    let (day, tm) = (pick(rng, DAYS), pick(rng, TIMES));
    vec![
        (
            format!("我想去{city}，{day}出发。"),
            format!("好的，去{city}。您想坐{transport}吗？"),
        ),
        (
            format!("{transport}吧，{tm}的。"),
            format!("查到了，{day}{tm}去{city}的{transport}还有票。要几张？"),
        ),
        (
            "一张。".into(),
            "好的，一张。请问要选靠窗还是靠过道？".into(),
        ),
        (
            "靠窗。".into(),
            "好的，靠窗座位。请您确认订单，支付后出票。".into(),
        ),
        ("确认。".into(), "出票成功，祝您旅途愉快。".into()),
    ]
}

fn medical(rng: &mut Rng) -> Vec<Turn> {
    let (sym, dep) = (pick(rng, SYMPTOMS), pick(rng, DEPARTMENTS));
    vec![
        (
            format!("你好，我{sym}。"),
            format!("您好，{sym}多久了？有没有发烧？"),
        ),
        (
            "两三天了，没有发烧。".into(),
            format!("建议挂{dep}看一下，先做个基础检查。"),
        ),
        (
            format!("{dep}现在还有号吗？"),
            "上午还有一个号，需要我帮您预约吗？".into(),
        ),
        (
            "需要，谢谢。".into(),
            format!("好的，已经帮您约好了，请到{dep}候诊。"),
        ),
        ("好的，谢谢。".into(), "不客气，注意多休息、多喝水。".into()),
    ]
}

fn tech_support(rng: &mut Rng) -> Vec<Turn> {
    let (prod, issue) = (pick(rng, PRODUCTS), pick(rng, ISSUES));
    vec![
        (
            format!("我买的{prod}{issue}。"),
            format!("抱歉给您添麻烦了。请问{prod}用了多久？"),
        ),
        (
            "大概一个月。".into(),
            format!("好的，先确认一下：重启之后还会{issue}吗？"),
        ),
        (
            "重启后会好一点，但过一会儿又出现。".into(),
            "明白。建议先升级固件，我可以把步骤发给你。".into(),
        ),
        (
            "好，麻烦发我。".into(),
            "已发送。如果升级后还出现，我们安排换新。".into(),
        ),
        ("好的，谢谢。".into(), "不客气，有问题随时联系。".into()),
    ]
}

fn shop(rng: &mut Rng) -> Vec<Turn> {
    let (prod, amt) = (pick(rng, PRODUCTS), pick(rng, AMOUNTS));
    vec![
        (
            format!("这个{prod}多少钱？"),
            format!("这款{prod}现在{amt}，今天有活动可以再减一点。"),
        ),
        (
            "有别的颜色吗？".into(),
            "有黑色和白色，黑色现货，白色要等两天。".into(),
        ),
        (
            "那要黑色的。".into(),
            format!("好的，黑色{prod}，需要开发票吗？"),
        ),
        (
            "需要，开公司抬头。".into(),
            "好的，请提供抬头信息，我这边登记。".into(),
        ),
        ("谢谢。".into(), "不客气，欢迎再来。".into()),
    ]
}

fn interview(rng: &mut Rng) -> Vec<Turn> {
    let topic = pick(rng, TOPICS);
    vec![
        (
            "你好，我来面试。".into(),
            "您好，请坐。先简单介绍一下自己吧。".into(),
        ),
        (
            format!("我做后端开发，平时也关注{topic}。"),
            "不错。能讲一个你解决过的难题吗？".into(),
        ),
        ("有一次线上延迟突然升高。".into(), "你是怎么定位的？".into()),
        (
            "先看监控，再逐步二分排查。".into(),
            "思路很清楚。你对团队协作有什么期待？".into(),
        ),
        (
            "希望有代码评审和技术分享。".into(),
            "我们的习惯正好是这样。今天先到这里，有消息会通知你。".into(),
        ),
    ]
}

fn casual(rng: &mut Rng) -> Vec<Turn> {
    let topic = pick(rng, TOPICS);
    vec![
        (
            "最近在忙什么？".into(),
            format!("在学{topic}，还在整理笔记。"),
        ),
        (
            format!("{topic}难吗？"),
            "入门不难，但想做出点东西要花时间。".into(),
        ),
        (
            "有什么建议吗？".into(),
            "先做一个小项目，遇到问题再补理论。".into(),
        ),
        ("有道理。".into(), "一起加油，有问题可以随时问我。".into()),
    ]
}

/// 生成 `dialogues` 段对话的完整语料文本（`用户：/助手：` 格式，段间空行分隔）。
///
/// 同一个 `seed` 产出的结果**逐字节相同**（Rng 是确定性的），便于复现与测试。
pub fn generate(dialogues: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let gens: [fn(&mut Rng) -> Vec<Turn>; 8] = [
        restaurant,
        bank,
        travel,
        medical,
        tech_support,
        shop,
        interview,
        casual,
    ];
    let mut out = String::with_capacity(dialogues * 160);
    for i in 0..dialogues {
        if i > 0 {
            out.push('\n'); // 段间空行（解析器把它当"新对话"）
        }
        let turns = gens[rng.choice(gens.len())](&mut rng);
        for (user, assistant) in turns {
            out.push_str(USER);
            out.push_str(&user);
            out.push('\n');
            out.push_str(ASSISTANT);
            out.push_str(&assistant);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个非空行都必须以角色标记开头，且用户/助手成对出现。
    #[test]
    fn test_generated_lines_have_alternating_roles() {
        let text = generate(50, 7);
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(!lines.is_empty());
        assert_eq!(lines.len() % 2, 0, "用户/助手必须成对");
        for (i, line) in lines.iter().enumerate() {
            if i % 2 == 0 {
                assert!(line.starts_with(USER), "第 {i} 行应是用户：{line}");
            } else {
                assert!(line.starts_with(ASSISTANT), "第 {i} 行应是助手：{line}");
            }
        }
    }

    /// 同种子逐字节可复现；不同种子必须不同。
    #[test]
    fn test_generation_is_deterministic_per_seed() {
        let a = generate(30, 42);
        let b = generate(30, 42);
        assert_eq!(a, b, "同种子必须完全一致");
        let c = generate(30, 43);
        assert_ne!(a, c, "不同种子应产出不同语料");
    }

    /// 对话段数 = 请求的段数（段间以空行分隔）。
    #[test]
    fn test_dialogue_count_matches_request() {
        for n in [1usize, 5, 40] {
            let text = generate(n, 1);
            let blocks: Vec<&str> = text
                .split("\n\n")
                .filter(|b| !b.trim().is_empty())
                .collect();
            assert_eq!(blocks.len(), n, "请求 {n} 段，实际 {}", blocks.len());
        }
    }

    /// 语料必须覆盖足够多的领域词与槽位（否则扩出来的语料仍然单一）。
    #[test]
    fn test_corpus_covers_domains_and_slots() {
        let text = generate(500, 42);
        for kw in [
            "订个位子",
            "高铁",
            "头疼",
            "面试",
            "最近在忙什么",
            "多少钱",
            "查一下余额",
        ] {
            assert!(text.contains(kw), "500 段里没有出现「{kw}」，领域覆盖不足");
        }
    }
}
