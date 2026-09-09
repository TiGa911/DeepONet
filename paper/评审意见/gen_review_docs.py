# -*- coding: utf-8 -*-
"""
gen_review_docs.py — 生成《文章修改意见》与《选稿意见》两份 Word 文档
参照 paper/相关文献/ 下两份样例文档的结构与风格，内容基于本项目论文
（paper/manuscript.tex, 2026-09-08 版）实际内容撰写。
"""
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

OUT_DIR = r'C:\Users\thunderobot\Desktop\py代码\fit\paper\评审意见'
PAPER_TITLE = 'DeepONet-Based Neural Network for Thomson Scattering Profile Fitting on EAST Tokamak'
PAPER_TITLE_ZH = '基于 DeepONet 的 EAST 汤姆逊散射剖面拟合神经网络方法'


def set_font(run, name='宋体', size=10.5, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    r = run._element
    r.rPr.rFonts.set(qn('w:eastAsia'), name)


def add_para(doc, text, size=10.5, bold=False, align=None, space_after=6,
             font='宋体', indent=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if indent is not None:
        p.paragraph_format.left_indent = Cm(indent)
    run = p.add_run(text)
    set_font(run, name=font, size=size, bold=bold)
    return p


def add_label_para(doc, label, text, size=10.5, space_after=6, indent=None):
    """带加粗前导标签的段落，如「优势：」+正文"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if indent is not None:
        p.paragraph_format.left_indent = Cm(indent)
    if label:
        r1 = p.add_run(label)
        set_font(r1, size=size, bold=True)
    r2 = p.add_run(text)
    set_font(r2, size=size)
    return p


def setup_doc(margins=2.0):
    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(margins)
        s.bottom_margin = Cm(margins)
        s.left_margin = Cm(margins)
        s.right_margin = Cm(margins)
    return doc


def fill_cell(cell, items, size=10.5):
    """items: list of (label, text, indent) 元组，label 可空"""
    cell.paragraphs[0].text = ''
    first = True
    for label, text, indent in items:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        if indent:
            p.paragraph_format.left_indent = Cm(indent)
        if label:
            r1 = p.add_run(label)
            set_font(r1, size=size, bold=True)
        r2 = p.add_run(text)
        set_font(r2, size=size)


def set_col_widths(table, widths_cm):
    for row in table.rows:
        for i, w in enumerate(widths_cm):
            if i < len(row.cells):
                row.cells[i].width = Cm(w)


# ================================================================
# 文档一：文章修改意见（审稿评估报告）
# ================================================================

JOURNALS = [
    ('Plasma Science and Technology (PST)',
     '约 1.7', '中科院三区', '3–6 周（一审约 1–2 个月）',
     'ASIPP 主办（IOP 出版），EAST 论文主场；编辑与审稿人熟悉 EAST 诊断语境，'
     '对"机器学习 + 聚变"交叉方向接受度好；合作作者臧庆为 EAST TS 诊断资深人员，与期刊读者群高度匹配。'
     '混合出版模式，传统订阅投稿无强制版面费。',
     'https://iopscience.iop.org/journal/1009-0630'),
    ('Fusion Engineering and Design (FED)',
     '约 1.9', '中科院三区', '2–4 周（一审约 2–3 个月）',
     'Elsevier 出版，偏工程应用导向；本文"端到端管线替代、面向大规模自动化数据处理"的叙事与 FED 定位匹配；'
     'EAST TVTS 诊断论文（Zhu et al. 2024）即发表于此。订阅制无强制版面费，可选 OA。',
     'https://www.sciencedirect.com/journal/fusion-engineering-and-design'),
    ('Plasma Physics and Controlled Fusion (PPCF)',
     '约 2.4', '中科院三区', '3–6 周（一审约 1.5–2.5 个月）',
     'IOP 出版；同类工作 Lan et al. 2023（CNN/BPNN 从线积分密度重建 EAST 密度剖面）发表于此，'
     '证明期刊对聚变诊断 ML 方法论文的接受度；"求稳"区间内影响因子偏高。无强制版面费。',
     'https://iopscience.iop.org/journal/0741-3335'),
    ('Review of Scientific Instruments (RSI)',
     '约 1.6', '中科院四区', '2–4 周（一审约 1–2 个月）',
     'AIP 出版，审稿效率以快著称；诊断数据处理与自动化工具类论文契合；'
     'EAST TS 诊断论文（Zang et al. 2011）发表于此。订阅制无强制版面费，OA 可选。',
     'https://pubs.aip.org/aip/rsi'),
    ('Journal of Instrumentation (JINST)',
     '约 1.3', '中科院四区', '3–5 周（一审约 1–2 个月）',
     'IOP/SISSA 出版，聚变与高能物理仪器、数据处理方法专门期刊，方法论型论文友好。无强制版面费。',
     'https://iopscience.iop.org/journal/1748-0221'),
    ('Scientific Reports',
     '约 3.9', '中科院三区（综合）', '2–3 周（一审约 2–3 个月）',
     'Springer Nature 出版，审稿快、录用相对宽松，为时间紧张时的兜底通道；'
     '注意：强制 OA 版面费（约 $2,590），且聚变领域认可度一般。',
     'https://www.nature.com/srep'),
]


def build_xiugai():
    doc = setup_doc()
    add_para(doc, '审稿评估报告', size=16, bold=True, font='黑体',
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para(doc, f'论文题目：{PAPER_TITLE}', size=12, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_para(doc, '（中文版：基于 DeepONet 的 EAST 汤姆逊散射剖面拟合神经网络方法）',
             size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    add_para(doc, '评估对象：paper/manuscript.tex（英文 17 页 / 中文 18 页，'
                  '6 图 3 表，23 篇参考文献，五架构：ProfileNet、LSTM、Transformer、'
                  'CNN-DeepONet、纯 CNN-1D 控制组）', size=9, space_after=10)

    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, t in enumerate(['项目', '评估内容']):
        hdr[i].paragraphs[0].text = ''
        r = hdr[i].paragraphs[0].add_run(t)
        set_font(r, size=11, bold=True, name='黑体')
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    rows = []

    # 1. 创新性
    rows.append(('研究方向\n创新性分析', [
        ('', '本文面向 EAST 托卡马克汤姆逊散射（TS）诊断的剖面拟合问题，提出基于 DeepONet '
              '（分支-主干算子学习）架构的神经网络方法，将现有 mtanh 拟合的七步手工管线'
              '（IQR 异常值清洗→芯部样条→台基 mtanh 参数优化→平滑过渡→左边界修正等）'
              '压缩为单次约 14 ms 前向传播。选题具有明确的工程驱动：EAST 年产数千炮放电，'
              '传统方法依赖诊断特异性代码分支与人工目视检验，难以支撑大规模自动化数据处理。', None),
        ('创新点 1：', '首次将 DeepONet 算子学习架构应用于 EAST TS 剖面拟合。'
              '与已有 ML 剖面工作相比——GPR（Choi & Poli, Phys. Plasmas 2022）、'
              'Diag2Diag TS 时序超分辨（Jalalvand et al., Nat. Commun. 2024）、'
              'CNN/BPNN 密度剖面重建（Lan et al., PPCF 2023）——本文的差异点在于'
              '受控的架构系统对比，而非单一模型的性能展示。', None),
        ('创新点 2：', '五架构受控对比设计严谨：四种架构（ProfileNet/LSTM/Transformer/'
              'CNN-DeepONet）共享同一 CoordDecoder 主干网络与物理约束损失，纯 CNN-1D '
              '作为不使用分支-主干结构的控制组；后补的 CNN-DeepONet 变体进一步消解了'
              '"编码器 vs 解码器"的实验混淆，将 CNN-1D 在 Ti 上的劣势干净地归因于解码器'
              '（换用主干后 Ti MAE 由 0.142 降至 0.082 keV，−42%）。这一对照设计在同类'
              '工作中少见，具有方法学示范意义。', None),
        ('创新点 3：', '消融实验得出核心发现"架构即正则化"：移除物理约束损失后，纯 CNN-1D '
              'MAE 恶化 +30.9%，而 DeepONet 架构反而改善（−11.5%/−11.4%），并以通用算子'
              '逼近定理（Lu et al. 2021）给出了数学解释——输出恒为光滑基函数的线性组合。'
              '该发现对科学机器学习领域的架构选择具有一般性启示。', None),
        ('创新点 4：', '提出诊断-架构匹配规律（Te→CNN、ne→Transformer、Ti→LSTM），'
              '并将其归因于三种诊断的空间采样特征差异；同时量化了剖面拟合差异对 LHW '
              '电流驱动计算的下游影响（ΔI≈24%），超出了纯拟合精度评估的范畴。', None),
        ('风险提示：', '"用 NN 拟合等离子体剖面"的概念本身已有先例（GPR 等），本文诚实定位为'
              '"mtanh 的快速近似"而非精度替代，创新性偏工程与方法学，属增量但扎实的贡献；'
              '审稿人可能质疑训练标签来自 mtanh 存在循环论证，论文已用 scatter-MSE 概念验证'
              '部分回应，建议在正文中进一步强化该论证。', None),
    ]))

    # 2. 整体内容
    rows.append(('文章整体\n内容分析', [
        ('优势：', '', None),
        ('1. ', '受控实验设计科学：统一 CoordDecoder、统一物理约束损失（λ 权重一致）、'
              '统一训练协议（AdamW + 余弦退火 + early stopping），保证五架构公平对比；'
              '按炮号分层 70/15/15 划分训练/验证/测试集，有效防止数据泄露。', None),
        ('2. ', '对照组体系完整：纯 CNN-1D（无分支-主干）与 CNN-DeepONet（CNN 编码器+共享主干）'
              '构成成对对照，实验结论可归因。', None),
        ('3. ', '训练数据规模与覆盖充分：161 炮、2015–2024 年跨十年（炮号 60150–149600），'
              'H 模样本 Te 342 / ne 302 / Ti 148。', None),
        ('4. ', '下游物理验证意识好：将 NN 剖面接入 LHW 电流驱动模型量化误差传播（ΔI≈24%），'
              '并明确指出该对比只量化"传播"而非"正确性"，表述克制、可信。', None),
        ('5. ', '论文定位诚实："fast approximation to mtanh"而非替代者，避免了夸大贡献'
              '的常见问题。', None),
        ('不足与提升空间：', '', None),
        ('1. ', '测试集规模偏小：仅 4 炮 20 个时间点（300 剖面），且全部为 2024 年炮号；'
              '跨年代泛化（80k–131k 早期炮）未验证，L 模仅 6 个时间点，跨模式结论的'
              '统计效力有限。建议补充 2–3 个早期炮号或注明限制。', None),
        ('2. ', '训练标签依赖 mtanh：NN 学习的是"mtanh 认为合理的剖面"，存在循环论证风险。'
              'scatter-MSE 概念验证实验（正文讨论中提及）是重要回应，但当前仅在讨论中'
              '一句话带过，建议扩展为正文小节或附表。', None),
        ('3. ', 'LHW 下游对比缺乏实验验证：无 LHCD 驱动电流的实验测量对照，且仅用了'
              '156400 一炮（n=1 案例），结论外推性有限。', None),
        ('4. ', 'ONETWO 输运验证未打通（ZFIT exit 91），完整物理闭环缺失；论文已如实说明，'
              '但这削弱了"下游影响"论证的深度。', None),
        ('5. ', '消融实验仅覆盖 Te 诊断的 4 个架构：CNN-DeepONet 未纳入消融（文中已注明'
              '留待未来工作）；LSTM 中间态（+12.7%）的解释偏定性。', None),
        ('6. ', '台基区域（ρ=0.85–1.0）无单独定量 MAE，仅定性讨论"NN 跟踪台基形状良好"，'
              '而台基恰是 H 模拟合的最难点。', None),
        ('7. ', '无不确定性量化：相比 GPR 方法的天然优势（输出后验不确定性）缺失，'
              '限制了在输运分析等下游任务中的可信度。', None),
        ('8. ', '正文声称计算了 RMSE、MaxAE、MeanRel% 等指标，但 Table 2 仅展示 MAE±std，'
              '承诺未兑现。', None),
        ('9. ', '消融实验未报告训练方差（是否多种子重复？），单次训练结果的 ±30.9%'
              '等数值的稳健性存疑。', None),
        ('10. ', 'Ti 训练样本仅 ~148（vs Te ~342），TXCS 稀疏输入（5–15 点）下各架构'
              '差异的结论受样本量限制。', None),
    ]))

    # 3. 语言表达
    rows.append(('文章语言表达\n可读性审查', [
        ('', '整体语言流畅、逻辑清晰、段落组织合理；经前期大幅缩减（42→16 页）后信息密度高，'
              '结构紧凑。以下问题建议修改：', None),
        ('1. ', '摘要第二段长句偏多：一段内同时承载消融发现、CNN-DeepONet 对照、诊断-架构'
              '匹配三件事，嵌套从句层级深（如 "demonstrating that the dot-product '
              'formulation ... providing structural regularization that replaces ..."），'
              '建议拆分为 2–3 句。', None),
        ('2. ', '结果 §3.5 与讨论 §4.1 内容重复度高："architecture as regularization"'
              '的论证两处大段重叠（同一组 +30.9%/−11.5% 数字与定理推导各写一遍），'
              '建议结果保留事实陈述、讨论聚焦机制解释与理论联系。', None),
        ('3. ', '术语一致性：CNN\\_Te、Transformer\\_ne 等代码式命名建议改为正式表述'
              '（如 "the CNN-based Te model"）；"ProfileNet" 与 "SetEncoder" 在文中混用，'
              '建议首次出现处统一定义；Ti 在摘要首次出现未给出全称（ion temperature），'
              '建议补全。', None),
        ('4. ', '部分表述强度与证据不匹配："fundamental architectural insight"、'
              '"fundamental implications for architecture selection in scientific ML"'
              '等措辞较自信，基于单一消融实验（Te、单次训练）下结论宜更审慎。', None),
        ('5. ', '个别数字表述需核对：正文"six distinct branches"（方法管线）与摘要/结论'
              '"seven-step pipeline"并存，建议统一并说明口径；"six-order-of-magnitude '
              'speedups"（FNO 引文）需核对原文表述。', None),
    ]))

    # 4. 标题
    rows.append(('标题合理\n性分析', [
        ('', '原标题：' + PAPER_TITLE, None),
        ('', '优点：方法（DeepONet）+ 对象（TS 剖面拟合）+ 装置（EAST）三要素齐全，'
              '信息准确、检索友好。', None),
        ('', '不足：未体现本文两项核心贡献——五架构受控对比与"架构即正则化"的消融发现；'
              '"Profile Fitting" 表述偏宽泛，未传达算子学习/受控实验的方法特色。', None),
        ('', '修改建议（按保留原风格程度排序）：', None),
        ('方案 A：', 'DeepONet-Based Profile Reconstruction for EAST Thomson Scattering: '
              'A Controlled Comparison of Five Neural Architectures with Physics-Constrained '
              'Learning（保留原名风格，突出对比与物理约束）', None),
        ('方案 B：', 'Branch–Trunk Architecture as Structural Regularization: A Systematic '
              'Neural Operator Study for Tokamak Kinetic Profile Fitting（突出核心发现，'
              '适合 ML 交叉类期刊）', None),
        ('方案 C：', 'Learning Operators for Tokamak Profile Reconstruction: '
              'Five Architectures Benchmarked against the mtanh Baseline on EAST'
              '（突出基准对比定位）', None),
    ]))

    # 5. 摘要
    rows.append(('摘要内容\n分析', [
        ('', '摘要基本涵盖目的、方法、结果、结论，并给出了关键定量数据（MAE 区间、'
              '+30.9%/−11.5%、Ti 0.142→0.082 keV、14 ms、<50 ms），质量较好。', None),
        ('1. ', '缺训练数据规模信息：建议补充"训练数据来自 2015–2024 年 161 炮放电、'
              '约 3400 个样本"一句，支撑方法可信度。', None),
        ('2. ', '缺与已有方法的定位句：建议增加一句与 GPR（Choi & Poli 2022）、'
              'Diag2Diag（2024）等工作的区分，说明本文的独特贡献在于受控架构对比'
              '而非概念首创。', None),
        ('3. ', '第二段信息过载，建议拆分（见语言审查第 1 条）。', None),
        ('4. ', '结论句可强化工程价值量化："完全自动化、单核推理 <20 ms，'
              '面向数千炮级批量处理"等。', None),
    ]))

    # 6. 引言
    rows.append(('引言内容\n分析', [
        ('', '引言"装置背景→mtanh 局限→ML 相关工作→DeepONet 动机→贡献清单"链条完整，'
              '六点贡献逐条对应正文内容，无空头承诺。', None),
        ('1. ', '相关工作仅一句话带过：GPR、Diag2Diag、Lan 三项工作各占半句，建议扩展为'
              '一段（或对比表），逐项说明"已有方法做了什么、缺什么"，凸显本文的受控对比'
              '与下游验证定位；IDA（Liu et al. 2024, Nucl. Fusion）等贝叶斯方法也应纳入'
              '比较视野。', None),
        ('2. ', '缺明确的研究问题/假设陈述：建议在贡献清单前增加一句可检验的研究问题，'
              '如"哪个架构组件（编码器/解码器/损失）决定拟合精度？DeepONet 分支-主干结构'
              '能否替代显式物理约束？"——这恰好是本文消融实验回答的问题，前置后论证更聚焦。', None),
        ('3. ', '"thousands of discharges annually"的数据规模主张需文献或内部数据支撑，'
              '否则建议弱化表述。', None),
        ('4. ', '"six distinct branches" 与 "seven-step pipeline" 口径不一，'
              '建议统一（见语言审查第 5 条）。', None),
    ]))

    # 7. 方法
    rows.append(('方法内容\n分析', [
        ('', '方法部分描述详实、可复现性好：数据构造（炮号分层划分、mtanh 标签生成、'
              'H98 模式分类）、四共享架构与 CNN 对照组的定义、物理约束损失逐项公式、'
              '训练超参数齐全。', None),
        ('1. ', '损失权重（λ1=0.05、λ2=0.05、λ3=0.01、w_log=0.1，区域加权 3×/25×）'
              '的选择依据缺失，建议补充一句超参数确定过程或敏感性分析说明。', None),
        ('2. ', '输入散点含 NaN/异常值的预处理协议未在方法中说明（讨论中提及"对 NaN '
              '敏感"），建议在 §2.2 明确：训练/推理时 NaN 剔除、通道截断等处理规则。', None),
        ('3. ', '推理速度测量协议缺失：应注明 CPU 型号、是否含数据搬运与后处理（PCHIP）'
              '耗时、重复次数，否则 "~14–19 ms" 不可复现。', None),
        ('4. ', 'mtanh 基线实现细节不足：基线是全部精度指标的参照系，应说明所用版本、'
              'IQR 阈值、台基位置（ρ=0.93）等关键参数，便于读者评估参照系质量。', None),
        ('5. ', '201 点输出网格与 PCHIP 后处理（强制单调+地板值）的交互未说明：'
              'PCHIP 是否会改变模型原始输出的误差统计？建议说明指标是在 PCHIP 之前'
              '还是之后计算。', None),
    ]))

    # 8. 结果与讨论
    rows.append(('结果与讨论\n内容分析', [
        ('结果部分：', '结构完整（拟合精度→H/L 模式→台基→峰值度→消融→LHW），'
              '表格规范、观察有据。问题如下：', None),
        ('1. ', '承诺指标未兑现：正文声称计算 RMSE/MaxAE/MeanRel%，Table 2 仅展示'
              'MAE±std；建议补充分模式 MeanRel% 列或附录表，尤其 H/L 对比中绝对 MAE '
              '受 156400 高 Te（最高 7.2 keV）影响，归一化指标更能支撑"相对误差可比"'
              '的论断。', None),
        ('2. ', '图编号管理混乱：源文件名为 fig2/fig4/fig5/fig6/fig7/fig9（fig3、fig8、'
              'fig10 删除后未清理），编译后 PDF 虽自动连续编号（Fig 1–7），但投稿上传'
              '时易出错，建议重命名源文件并清理 paper_results/figures/ 下的遗留文件'
              '（fig7_ablation.png、fig8_architecture_table.png 等未使用文件）。', None),
        ('3. ', 'LHW 案例为单炮（156400），建议明确标注案例性质并弱化外推表述。', None),
        ('讨论部分：', '问题如下：', None),
        ('4. ', '§4.1 与 §3.5 大段重复（见语言审查第 2 条），建议合并或重构：结果给事实、'
              '讨论给机制——可深入"为什么主干正则化对稀疏诊断（TXCS 5–15 点）有效而对'
              '密集诊断（TS 20–30 点）冗余"的理论解释，并与 GPR 核平滑做类比。', None),
        ('5. ', '"诊断-架构匹配"论证有力但缺一般性检验：建议在合成数据上控制通道密度/'
              '均匀性验证"稀疏性→主干收益"的假设，或作为 future work 明确提出。', None),
        ('6. ', '未来工作列举 5 条（a–e）偏多，建议聚焦 2–3 条并给出优先级。', None),
    ]))

    # 9. 结论
    rows.append(('结论内容\n分析', [
        ('', '结论五点结构清晰，与贡献清单一一对应。', None),
        ('1. ', '与摘要重复度高：同一批数字（0.36–0.53 keV、+30.9%、−42%、24%）在'
              '摘要、结果、结论出现三次，建议结论减少逐条复述，突出升华——'
              '"首次 + 工程价值"（数千炮无人干预处理能力）与"架构即正则化"的一般性启示。', None),
        ('2. ', '建议补充一句对政策/工程实践的启示（如 EAST 自动化数据流水线的部署路径），'
              '提升结论的落点。', None),
    ]))

    # 10. 图片
    rows.append(('图片内容\n分析', [
        ('', '论文共 6 图：Fig 1 架构示意（TikZ 矢量图，已单独绘制）、Fig 2 H 模典型叠加、'
              'Fig 4 台基放大、Fig 5 峰值度散点、Fig 6 MAE 箱线图、Fig 7 H/L 对比、'
              'Fig 9 LHW 四面板（编译后自动编号 1–7）。', None),
        ('1. ', '黑白印刷可读性：Fig 2 六条曲线叠加（mtanh+5 架构），Fig 5 散点按炮号'
              '着色，Fig 7 柱状图 H/L 分组——PST/FED 等期刊纸质版多为黑白，建议检查'
              '线型/填充样式/形状标记的差异化，而非仅靠颜色区分。', None),
        ('2. ', 'Fig 6 箱线图建议在纵轴或图注注明每箱样本量（20 时间点/诊断）。', None),
        ('3. ', 'Fig 9 四面板单位与量级：A/cm² 上标、MW/m³ 等符号在导出 PNG 后清晰度'
              '需检查；四面板字体大小应一致。', None),
        ('4. ', '所有图片已有 300 dpi，符合投稿要求；Fig 1 为矢量 PDF，投稿时注意期刊'
              '对 TikZ/PDF 图件的兼容要求（IOP 要求 EPS/PDF，Elsevier 要求 300 dpi '
              '位图或矢量均可）。', None),
        ('5. ', '遗留文件清理：fig3_lmode_overlay.png、fig10_temporal.png、'
              'fig7_ablation.png、fig8_architecture_table.png 等未使用文件建议移出'
              'figures 目录，避免投稿误传。', None),
    ]))

    # 11. 表格
    rows.append(('表格内容\n分析', [
        ('Table 1（测试集）: ', '信息完整，156010 H98=0 的脚注处理得当；建议增加'
              '"与训练集时间间隔"一列，凸显测试炮号（2024）与训练数据（至 149600 炮）'
              '的独立性与时域泛化意义。', None),
        ('Table 2（拟合精度）: ', '5 架构×3 诊断×3 模式，信息量大但列数多（7 列），'
              '字号偏小时可读性受限；建议：① 补充 MeanRel% 等承诺指标；② "All/H/L"'
              '行的最优值粗体标注已做，规范。', None),
        ('Table 3（消融）: ', '仅 Te 诊断 4 架构；建议标注训练种子/重复次数，'
              '如有多种子结果报告均值±std；"improved/degraded" 标注直观。', None),
        ('', '所有表格均在正文中有明确引用（\\ref），符合规范。', None),
    ]))

    # 12. 参考文献
    rows.append(('参考文献\n内容分析', [
        ('', '共 23 篇，对原创研究论文偏少（同类聚变 ML 论文一般 30–45 篇），'
              '且存在引用规范问题，需投稿前逐条核对。', None),
        ('1. ', 'LHW 章节（§3.6）全文零引用：LHCD 实验背景（Li et al. 2021, Nucl. '
              'Fusion）与参量衰变（Li et al. 2024）两条已定义的 bibitem'
              '（lhcd_east、lh_pdi_east）未在正文引用，建议在 §3.6 首段补充。', None),
        ('2. ', '4 条 bibitem 定义后未在正文引用：ida_east（Bayesian IDA）、lhcd_east、'
              'lh_pdi_east、pinn_review——要么在正文相应位置引用（引言相关工作段引用'
              'ida_east；损失函数段引用 pinn_review），要么删除条目。', None),
        ('3. ', '元数据错误（经 CrossRef/OpenAlex 核查发现的已知问题）：gpr_east、'
              'diag2diag、lhcd_east 等条目的标题/卷期与数据库记录不匹配；east_overview '
              '2017 与 2019 版本待确认；lan_cnn_east 缺 DOI；pinn_east 作者署名不规范'
              '（以机构名代替作者、会议信息不完整）；deeponet_equilibrium 为会议论文'
              '（IAEA FEC），需按目标期刊格式规范。', None),
        ('4. ', '格式统一：投稿前按目标期刊模板重排（PST/PPCF 用 IOP 数字编号格式，'
              'FED 用 Elsevier 格式），确保期刊缩写、卷期页码、DOI 齐全，'
              '所有引用在正文中均有对应。', None),
    ]))

    # 13. 期刊推荐
    rows.append(('期刊推荐内容\n（按求稳+快速\n投稿要求推荐）', [
        ('', '以下推荐按"初审 2 个月内、中科院三区/四区可接受、版面费无限制"的投稿要求排序。'
              '影响因子与分区为 2024–2025 年度约值，投稿前请在最新 JCR 与中科院文献情报中心'
              '《期刊分区表》核实；所荐期刊均不在中科院《国际期刊预警名单》内。', None),
    ]))
    for i, (name, ifv, zone, period, reason, url) in enumerate(JOURNALS, 1):
        rows.append((f'推荐 {i}：\n{name}', [
            ('影响因子：', ifv, None),
            ('中科院分区：', zone, None),
            ('审稿周期：', period, None),
            ('推荐理由：', reason, None),
            ('期刊网址：', url, None),
        ]))

    # 填充表格
    for sec_name, items in rows:
        row = table.add_row()
        c0, c1 = row.cells
        c0.paragraphs[0].text = ''
        r = c0.paragraphs[0].add_run(sec_name)
        set_font(r, size=10, bold=True, name='黑体')
        c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        fill_cell(c1, items)
    set_col_widths(table, [3.2, 13.8])

    doc.save(OUT_DIR + r'\文章修改意见.docx')
    print('saved: 文章修改意见.docx')


# ================================================================
# 文档二：选稿意见
# ================================================================

def build_xuangao():
    doc = setup_doc()

    add_para(doc, f'论文《{PAPER_TITLE}》选稿意见', size=15, bold=True,
             font='黑体', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)

    add_label_para(doc, '文章主题：',
                   '本研究面向 EAST 托卡马克汤姆逊散射（TS）诊断的剖面拟合问题，'
                   '提出基于 DeepONet（分支-主干算子学习）的神经网络方法，'
                   '将传统 mtanh 七步手工拟合管线压缩为单次约 14 ms 前向传播。'
                   '在 4 炮 20 个时间点（300 个剖面）的未见测试集上系统对比了五种架构'
                   '（ProfileNet、LSTM、Transformer、CNN-DeepONet 共享同一主干解码器，'
                   '纯 CNN-1D 作为控制组），并通过物理约束消融实验揭示核心发现：'
                   'DeepONet 分支-主干点积结构自带函数空间正则化（去约束后纯 CNN 恶化 '
                   '+30.9%，DeepONet 架构反而改善 −11.5%）；CNN-DeepONet 对照进一步'
                   '将 CNN-1D 在 Ti 上的劣势归因于解码器（换用主干后 Ti MAE 由 0.142 '
                   '降至 0.082 keV，−42%）。下游 LHW 电流驱动计算显示 NN 剖面与 mtanh '
                   '基线偏差约 24%。该研究在聚变诊断与科学机器学习交叉方向具有较强'
                   '创新性，尤其是受控架构对比与消融实验的方法学贡献。', size=10.5)

    # 一、写作与内容改进建议
    add_para(doc, '一、写作与内容改进建议（投稿合规性）', size=12, bold=True,
             font='黑体', space_after=6)
    items1 = [
        '补齐基金资助信息：Acknowledgments 现为占位符"[funding information to be added]"，'
        'IOP/Elsevier 期刊均要求列出基金号；若无基金资助也需明确说明，不可留占位符。',
        '补充数据可用性声明（Data Availability Statement）：EAST 实验数据由 ASIPP 管控，'
        '需按期刊模板声明数据获取途径（如"数据可根据合理请求向 ASIPP 获取"），'
        'IOP 系期刊为强制要求。',
        '完善作者信息：CRediT 作者贡献声明、通讯作者标注、ORCID（IOP/Elsevier 均要求）。',
        '补充利益冲突声明（Competing Interests）：无冲突也需显式声明。',
        '伦理声明：本工作为实验数据分析、不涉及人体/动物实验，通常无需伦理批号，'
        '但需在投稿系统中确认对应栏目填写。',
        '参考文献格式转换：PST/PPCF 按 IOP 数字编号格式、FED 按 Elsevier 格式重排；'
        '当前存在 4 条未引用条目（ida_east、lhcd_east、lh_pdi_east、pinn_review）与'
        'LHW 章节零引用问题（详见《文章修改意见》参考文献部分），需先修整。',
        '图件规范：期刊一般要求单独上传 300 dpi 以上图件；Fig 1 为 TikZ 矢量图，'
        '导出 PDF/EPS 时注意字体嵌入；检查黑白印刷可读性（Fig 2 六线叠加、Fig 5 按炮'
        '着色，建议增加线型/标记差异化）。',
        '英文润色：语言总体流畅，但摘要长句与个别表述（如代码式命名 CNN\\_Te）建议'
        '专业润色后投稿。',
        '预印本：arXiv 预印本不影响 IOP/Elsevier 期刊投稿（均为常规单盲/双盲评审，'
        '无预印本禁令），可先挂 arXiv 抢占时间戳。',
        'Cover letter 重点：突出"EAST 数千炮级数据自动化处理"的应用价值与受控'
        '对比实验的方法学贡献，与期刊 Aims & Scope 对齐（PST 强调等离子体科学与'
        '技术应用）。',
    ]
    for i, t in enumerate(items1, 1):
        add_para(doc, f'{i}. {t}', size=10.5, space_after=4)

    # 二、选刊意见
    add_para(doc, '二、选刊意见', size=12, bold=True, font='黑体', space_after=6)
    add_label_para(doc, '投稿要求：',
                   '影响因子无硬性要求（中科院三区/四区均可，求稳）；审稿周期初审 2 个月内、'
                   '总体不超过 6 个月；版面费无限制（OA 可选）；避开中科院《国际期刊预警名单》'
                   '与科睿唯安标记 "On Hold" 的期刊；无被拒记录期刊需避开；期刊须有正式'
                   '出版刊号、SCI 检索（不可为 ESCI 或增刊）。', size=10.5)
    add_para(doc, '注：以下影响因子与分区为 2024–2025 年度约值，投稿前请在最新 JCR '
                  '与中科院文献情报中心《期刊分区表》核实；所有推荐期刊均为 SCI 收录、'
                  '不在预警名单内。', size=9, space_after=8)

    # 表格
    headers = ['期刊名称', '编号', '中科院分区 / IF', '推荐原因', '优点', '缺点',
               '审稿周期', '篇幅要求']
    table = doc.add_table(rows=1, cols=8)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    for i, t in enumerate(headers):
        hdr[i].paragraphs[0].text = ''
        r = hdr[i].paragraphs[0].add_run(t)
        set_font(r, size=9, bold=True, name='黑体')
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    journal_rows = [
        ('Plasma Science and Technology (PST)', '1',
         '中科院三区\nIF 约 1.7',
         'ASIPP 主办（IOP 出版），EAST 论文主场；编辑与审稿人熟悉 EAST 诊断语境；'
         '期刊设有 AI/机器学习与聚变交叉方向，与本文主题高度匹配。',
         '① SCI 稳定收录，混合模式无强制版面费；② 审稿快（初审 3–6 周），符合快速诉求；'
         '③ 领域匹配度高，命中率高；④ 合作作者臧庆为 EAST TS 诊断资深人员，'
         '编辑部熟悉度好。',
         '① 影响因子在推荐列表中偏低；② 对"应用价值"要求明确，'
         '需在 cover letter 突出工程落地性；③ 篇幅限制相对严格。',
         '初审约 3–6 周，一审约 1–2 个月；修改后接收约 2–4 个月。',
         '研究论文一般 6000–9000 词；图 6–10 张（本文 6 图 3 表，符合）。'),
        ('Plasma Physics and Controlled Fusion (PPCF)', '2',
         '中科院三区\nIF 约 2.4',
         'IOP 出版；同类工作 Lan et al. 2023（CNN/BPNN 从线积分密度重建 EAST 密度剖面）'
         '发表于此，证明期刊对聚变诊断 ML 方法论文的接受度；"求稳"区间内 IF 最高。',
         '① SCI 稳定收录，无强制版面费；② IF 2.4 高于 PST/FED，'
         '毕业后认可度更好；③ 同类论文先例明确。',
         '① 审稿要求比 PST 更严，对方法学严谨性（如消融实验统计稳健性）'
         '有较高要求；② 竞争稿源多。',
         '初审约 3–6 周，一审约 1.5–2.5 个月。',
         '一般 5000–8000 词，图表适量。'),
        ('Fusion Engineering and Design (FED)', '3',
         '中科院三区\nIF 约 1.9',
         'Elsevier 出版，偏工程应用导向；本文"端到端管线替代、面向数千炮级自动化'
         '数据处理"的叙事与 FED 定位匹配；EAST TVTS 诊断论文（Zhu et al. 2024）'
         '即发表于此。',
         '① SCI 稳定收录，订阅制无强制版面费；② 工程自动化叙事契合度高；'
         '③ Elsevier 投稿系统流程规范。',
         '① 一审周期稍长（约 2–3 个月）；② 对物理深度要求相对低但要求'
         '"工程可用性"证据充分（当前 ONETWO 链路未打通需在文中交代）。',
         '初审约 2–4 周，一审约 2–3 个月。',
         '无严格字数限制，一般 5000–8000 词。'),
        ('Review of Scientific Instruments (RSI)', '4',
         '中科院四区\nIF 约 1.6',
         'AIP 出版，诊断数据处理与自动化工具类论文契合；EAST TS 诊断论文'
         '（Zang et al. 2011）发表于此；审稿效率以快著称。',
         '① 审稿快（初审 2–4 周），符合快速诉求；② 命中率高、稳妥；'
         '③ 无强制版面费。',
         '① 影响因子低（四区），认可度有限；② 偏仪器类，需在投稿时强调'
         '"方法学 + 自动化工具"属性而非纯物理结论。',
         '初审约 2–4 周，一审约 1–2 个月；整体约 2–3 个月。',
         '一般 4000–6000 词；可附补充材料放训练细节。'),
        ('Journal of Instrumentation (JINST)', '5',
         '中科院四区\nIF 约 1.3',
         'IOP/SISSA 出版，聚变与高能物理仪器、数据处理方法专门期刊，'
         '方法论型论文友好。',
         '① 无强制版面费；② 审稿较快；③ 主题契合（仪器数据自动化处理）。',
         '① 影响因子最低；② 在聚变物理圈引用面偏窄。',
         '初审约 3–5 周，一审约 1–2 个月。',
         '一般 5000–8000 词。'),
        ('Scientific Reports', '6',
         '中科院三区（综合）\nIF 约 3.9',
         'Springer Nature 出版；审稿快、录用相对宽松，为时间紧张时的兜底通道；'
         '版面费无限制前提下可选。',
         '① 审稿最快（初审 2–3 周）；② IF 3.9 为列表中最高；'
         '③ 多学科读者面广，曝光度高。',
         '① 强制 OA 版面费约 $2,590；② 聚变领域认可度一般，'
         '存在"灌水"口碑风险；③ 需承担公开评审压力。',
         '初审约 2–3 周，一审约 2–3 个月。',
         '无字数上限，一般 5000–8000 词。'),
    ]

    for row_data in journal_rows:
        row = table.add_row()
        for i, t in enumerate(row_data):
            row.cells[i].paragraphs[0].text = ''
            r = row.cells[i].paragraphs[0].add_run(t)
            set_font(r, size=8.5, bold=(i == 0))
    set_col_widths(table, [2.4, 0.9, 2.0, 3.4, 3.3, 3.2, 2.0, 2.2])

    # 三、推荐排序与建议
    add_para(doc, '', space_after=2)
    add_para(doc, '三、推荐排序与建议', size=12, bold=True, font='黑体', space_after=6)

    add_label_para(doc, '首选：', 'Plasma Science and Technology（PST，排名第 1）', size=10.5)
    add_para(doc, '中科院三区 / IF 约 1.7。ASIPP 主办、EAST 论文主场，审稿快（初审 3–6 周），'
                  '混合模式无强制版面费，领域匹配度与命中率均为最高。投稿时 cover letter '
                  '突出"数千炮级数据自动化处理"的工程价值与五架构受控对比的方法学贡献。',
             size=10.5, indent=0.6)

    add_label_para(doc, '次选：', 'Plasma Physics and Controlled Fusion（PPCF，排名第 2）', size=10.5)
    add_para(doc, '中科院三区 / IF 约 2.4。同类论文（Lan et al. 2023）发表于此，接受度有'
                  '先例；IF 在"求稳"区间内最高，对毕业/评奖更有利。若被 PST 拒稿，'
                  '根据审稿意见完善消融统计稳健性后转投。', size=10.5, indent=0.6)

    add_label_para(doc, '备选 1：', 'Fusion Engineering and Design（FED，排名第 3）', size=10.5)
    add_para(doc, '中科院三区 / IF 约 1.9。工程自动化叙事匹配，一审周期稍长（2–3 个月），'
                  '适合不急于毕业时的第二顺位。', size=10.5, indent=0.6)

    add_label_para(doc, '备选 2：', 'Review of Scientific Instruments（RSI，排名第 4）', size=10.5)
    add_para(doc, '中科院四区 / IF 约 1.6。审稿最快（初审 2–4 周）、命中率高，'
                  '是"求稳+快速"诉求下最稳妥的选择；投 RSI 时建议强调方法学与自动化工具'
                  '属性。', size=10.5, indent=0.6)

    add_label_para(doc, '备选 3：', 'Journal of Instrumentation（JINST，排名第 5）', size=10.5)
    add_para(doc, '中科院四区 / IF 约 1.3。方法论型论文友好，审稿较快，'
                  '作为四区兜底选项。', size=10.5, indent=0.6)

    add_label_para(doc, '兜底：', 'Scientific Reports（排名第 6）', size=10.5)
    add_para(doc, '中科院三区（综合）/ IF 约 3.9。审稿最快但强制 OA 版面费（约 $2,590，'
                  '已确认可接受）；聚变圈认可度一般，仅建议在时间极紧且前序期刊均拒稿时'
                  '使用。', size=10.5, indent=0.6)

    add_para(doc, '', space_after=2)
    add_label_para(doc, '投稿顺序：',
                   '基于"求稳 + 初审 2 个月内"的要求，建议按 PST → PPCF → FED → RSI → '
                   'Scientific Reports 依次投稿。PST 与 PPCF 均为 IOP 系统，拒稿后可快速'
                   '迁移（部分 IOP 期刊支持快速转投）。RSI 为最稳妥的快速通道。'
                   '无论投哪家，请先完成《文章修改意见》中标注为"投稿前必修"的事项：'
                   '补齐基金与数据声明、处理未引用文献、参考文献元数据核对与格式转换。',
             size=10.5)

    # 四、补充推荐：中文核心与普刊
    add_para(doc, '', space_after=2)
    add_para(doc, '四、补充推荐：中文核心与普刊', size=12, bold=True, font='黑体', space_after=6)
    add_para(doc, '以下补充中文期刊选项（投稿要求同等适用：快速、版面费无限制）。核心级别以最新'
                  '《中文核心期刊要目总览》（2023 年版）与 CSCD 目录为准，投稿前请核实；所有中文期刊'
                  '均要求中文摘要+英文摘要、中图分类号与作者简介。', size=10.5)
    add_label_para(doc, '⚠ 重复发表提醒：',
                   '同一研究内容中英文双语发表存在重复发表（一稿多投）风险——中文期刊投稿声明均要求'
                   '"未在国内外公开刊物发表"。若英文版（PST/FED）已投出或计划投出，中文版需以显著不同'
                   '的角度/内容成文（如侧重方法细节与数据集、补充新实验），或与导师确认；稳妥顺序为'
                   '英文版拒稿后再投中文，或先投中文再投英文。', size=10.5)

    # （一）中文核心
    add_para(doc, '（一）中文核心期刊', size=11, bold=True, font='黑体', space_after=6)
    headers_cn = ['期刊名称', '编号', '级别', '推荐原因', '优点', '缺点', '审稿周期', '篇幅要求']
    table2 = doc.add_table(rows=1, cols=8)
    table2.style = 'Table Grid'
    hdr2 = table2.rows[0].cells
    for i, t in enumerate(headers_cn):
        hdr2[i].paragraphs[0].text = ''
        r = hdr2[i].paragraphs[0].add_run(t)
        set_font(r, size=9, bold=True, name='黑体')
        hdr2[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    cn_core_rows = [
        ('核聚变与等离子体物理', '1',
         '北大核心 / CSCD\n（核工业西南物理研究院主办）',
         '国内唯一中文聚变专业期刊，EAST/HL-2A 等装置等离子体物理与诊断方法论文的稳定发稿地，'
         '与本文主题最对口的中文期刊。',
         '① 领域完全对口，审稿人熟悉聚变诊断语境；② 中文撰写，写作与修改成本低；'
         '③ 编辑部对 EAST 背景无需铺垫。',
         '① 中文期刊在国际学术评价中权重较低；② 版面费约 1500–3000 元；'
         '③ 周期比普刊长。',
         '初审约 1–2 个月，一审约 2–4 个月。',
         '5000–8000 字，图表适量。'),
        ('计算物理', '2',
         '北大核心 / CSCD\n（北京应用物理与计算数学研究所主办）',
         'ML+物理计算交叉方向直接对口——神经网络代理模型、物理约束学习等内容与该刊定位高度契合。',
         '① 主题匹配度最高的核心期刊之一（计算方法类论文主场）；② 认可度较好。',
         '① 审稿人对聚变诊断背景可能不熟，需花篇幅铺垫；② 对算法创新性要求较高，'
         '纯应用型论文命中率一般。',
         '初审约 1–2 个月，一审约 2–3 个月。',
         '5000–8000 字。'),
        ('强激光与粒子束', '3',
         '北大核心 / CSCD\n（中国工程物理研究院主办）',
         '覆盖等离子体物理、粒子束与诊断技术，接受数值模拟与 ML 应用类论文。',
         '① 审稿相对较快；② 编辑部规模大、流程规范。',
         '① 偏强激光/束流方向，托卡马克剖面拟合主题稍偏；② 需在引言中强化诊断应用背景。',
         '初审约 2–4 周，一审约 1–2 个月。',
         '6000–9000 字。'),
        ('核技术', '4',
         '北大核心 / CSCD\n（中科院上海应用物理研究所主办）',
         '核电子学、测量与数据处理方法类论文对口（TS 诊断数据自动化处理属该刊收稿范围）。',
         '① 方法学论文友好；② 审稿规范、反馈细致。',
         '① 与"托卡马克剖面拟合"主题匹配度一般，需以"诊断数据处理方法"角度切入。',
         '初审约 1–2 个月，一审约 2–3 个月。',
         '5000–8000 字。'),
        ('物理学报', '5',
         '北大核心 / CSCD / SCI 收录\n（中国物理学会主办）',
         '中文物理学最高水平期刊，若中文版以"物理约束学习+算子学习理论"为主线可作冲刺选项。',
         '① 中文期刊中认可度最高；② SCI 检索，兼顾国际评价。',
         '① 要求高、命中率低；② 周期长（一审 2–4 个月）；③ 与"求稳快速"诉求不符，'
         '仅建议作为冲刺。',
         '初审约 1–2 个月，一审约 2–4 个月。',
         '6000–10000 字。'),
    ]
    for row_data in cn_core_rows:
        row = table2.add_row()
        for i, t in enumerate(row_data):
            row.cells[i].paragraphs[0].text = ''
            r = row.cells[i].paragraphs[0].add_run(t)
            set_font(r, size=8.5, bold=(i == 0))
    set_col_widths(table2, [2.4, 0.9, 2.0, 3.4, 3.3, 3.2, 2.0, 2.2])

    # （二）普刊
    add_para(doc, '', space_after=2)
    add_para(doc, '（二）普刊（非核心，快速兜底）', size=11, bold=True, font='黑体', space_after=6)
    table3 = doc.add_table(rows=1, cols=8)
    table3.style = 'Table Grid'
    hdr3 = table3.rows[0].cells
    for i, t in enumerate(headers_cn):
        hdr3[i].paragraphs[0].text = ''
        r = hdr3[i].paragraphs[0].add_run(t)
        set_font(r, size=9, bold=True, name='黑体')
        hdr3[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    cn_gen_rows = [
        ('安徽理工大学学报（自然科学版）', '1',
         '普通学报（非核心，本校学报）',
         '第一署名单位本校学报，理工科综合收稿；对研究生毕业条件与在校成果认定最友好，'
         '是"普刊"诉求下最实用的选择。',
         '① 审稿最快（1–4 周）；② 命中率接近确定；③ 版面费低（约 500–1000 元）；'
         '④ 毕业/评奖时"本校学报"通常被认可。',
         '① 学术认可度最低；② 不属于任何核心目录。',
         '初审约 1–2 周，录用约 1 个月。',
         '4000–6000 字。'),
        ('核科学与技术（汉斯出版社）', '2',
         'OA 普刊（非核心）',
         '主题名实相符的核科学 OA 期刊，收稿范围宽（核技术、等离子体、数值模拟均可）。',
         '① 审稿极快（1–2 周）；② 知网可检索。',
         '① 学术认可度低，无评奖/毕业加分量；② 版面费约 1000–2000 元。',
         '初审约 1 周，录用约 2–4 周。',
         '不限。'),
        ('现代物理（汉斯出版社）', '3',
         'OA 普刊（非核心）',
         '物理学综合 OA 期刊，可收聚变与 ML 交叉方向稿件，兜底中的兜底。',
         '① 审稿极快；② 发表门槛最低。',
         '① 认可度低；② 版面费约 1000–2000 元；③ 与专业期刊相比引用面窄。',
         '初审约 1 周，录用约 2–4 周。',
         '不限。'),
    ]
    for row_data in cn_gen_rows:
        row = table3.add_row()
        for i, t in enumerate(row_data):
            row.cells[i].paragraphs[0].text = ''
            r = row.cells[i].paragraphs[0].add_run(t)
            set_font(r, size=8.5, bold=(i == 0))
    set_col_widths(table3, [2.4, 0.9, 2.0, 3.4, 3.3, 3.2, 2.0, 2.2])

    # 中文投稿顺序
    add_para(doc, '', space_after=2)
    add_label_para(doc, '中文投稿顺序：',
                   '若英文版受阻或需要中文成果：首选《核聚变与等离子体物理》（最对口核心）→ 次选'
                   '《计算物理》或《强激光与粒子束》→ 求稳快速选《安徽理工大学学报（自然科学版）》'
                   '→ 兜底汉斯普刊。注意：无论投哪家中文期刊，须先确认英文版投稿状态，避免重复发表；'
                   '中文稿件需补充中图分类号与第一作者简介。', size=10.5)

    doc.save(OUT_DIR + r'\选稿意见.docx')
    print('saved: 选稿意见.docx')


if __name__ == '__main__':
    build_xiugai()
    build_xuangao()
    print('done')
