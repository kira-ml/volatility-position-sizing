#!/usr/bin/env python3
"""
generate_linkedin_research_brief.py

Generates a publication-quality, 2-page Research Brief for LinkedIn.
Serious, technical, dense with evidence, and completely ego-free.
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

OUTPUT_PDF = 'outputs/paper/linkedin_research_brief.pdf'

FIGURE_PATHS = {
    'cumulative_returns': 'outputs/figures/03_cumulative_returns.png',
    'experiment_results': 'outputs/figures/09_experiment_results.png',
    'model_comparison': 'outputs/figures/05_model_comparison.png',
    'mincer_zarnowitz': 'outputs/figures/02_mincer_zarnowitz.png',
}

# -----------------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------------

def create_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='BriefTitle',
        parent=styles['Heading1'],
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=0.05*inch,
        fontName='Times-Bold'
    ))

    styles.add(ParagraphStyle(
        name='BriefSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=0.15*inch,
        fontName='Times-Italic'
    ))

    styles.add(ParagraphStyle(
        name='BriefSection',
        parent=styles['Heading2'],
        fontSize=13,
        alignment=TA_LEFT,
        spaceAfter=0.05*inch,
        spaceBefore=0.1*inch,
        fontName='Times-Bold'
    ))

    styles.add(ParagraphStyle(
        name='BriefSubSection',
        parent=styles['Heading3'],
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=0.02*inch,
        fontName='Times-Bold'
    ))

    styles.add(ParagraphStyle(
        name='BriefBody',
        parent=styles['Normal'],
        fontSize=9.5,
        alignment=TA_JUSTIFY,
        spaceAfter=0.04*inch,
        firstLineIndent=0.1*inch,
        fontName='Times-Roman'
    ))

    styles.add(ParagraphStyle(
        name='BriefTableCaption',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        spaceAfter=0.05*inch,
        fontName='Times-Italic'
    ))

    styles.add(ParagraphStyle(
        name='BriefFooter',
        parent=styles['Normal'],
        fontSize=8.5,
        alignment=TA_CENTER,
        spaceBefore=0.1*inch,
        fontName='Times-Italic'
    ))

    return styles

# -----------------------------------------------------------------------------
# BUILD FUNCTIONS
# -----------------------------------------------------------------------------

def build_content(styles, story):
    """Build the 2-page Research Brief content."""
    
    # --- HEADER ---
    story.append(Paragraph("Volatility Forecasting for Dynamic Position Sizing", styles['BriefTitle']))
    story.append(Paragraph("A Machine Learning Approach to Forward Volatility Estimation", styles['BriefSubtitle']))
    story.append(Paragraph("Python • LightGBM • Scikit-learn • ReportLab", styles['BriefSubtitle']))
    story.append(Spacer(1, 0.15*inch))

    # --- SECTION 1: PROBLEM & METHODOLOGY ---
    story.append(Paragraph("1. Problem & Methodology", styles['BriefSection']))
    
    story.append(Paragraph(
        "This research brief explores a supervised learning approach to forecasting 5-day forward realized volatility "
        "for individual equities. The forecast is employed to drive a dynamic position-sizing rule where "
        "position_scale = target_vol / predicted_vol. The objective is to improve risk-adjusted performance by "
        "allocating capital inversely proportional to predicted market turbulence.",
        styles['BriefBody']
    ))
    
    story.append(Paragraph(
        "A baseline-first framework was employed to justify model complexity. Three progressively sophisticated baselines "
        "were established: (1) 21-day rolling historical volatility, (2) EWMA volatility (λ=0.94, RiskMetrics standard), "
        "and (3) Ridge-regularized linear regression. LightGBM was evaluated as the advanced candidate. "
        "All models were assessed using purged walk-forward cross-validation (5 splits, 252-day test windows, 5-day embargo) "
        "to prevent look-ahead bias.",
        styles['BriefBody']
    ))

    story.append(Spacer(1, 0.1*inch))

    # --- SECTION 2: EXPERIMENTAL RESULTS ---
    story.append(Paragraph("2. Experimental Results", styles['BriefSection']))

    # --- Subsection 2.1: Model Performance ---
    story.append(Paragraph("2.1 Model Performance", styles['BriefSubSection']))
    
    table_data = [
        ['Feature Set', 'Best Model', 'RMSE', 'MAE', 'MZ β', 'MZ p-value'],
        ['Baseline 1 (Rolling)', 'LightGBM', '0.127', '0.100', '0.395', '0.001'],
        ['Baseline 2 (EWMA)', 'Ridge', '0.160', '0.121', '3.841', '~0.000'],
        ['Baseline 3 (Mixed)', 'LightGBM', '0.119', '0.094', '0.711', '0.141'],
        ['Advanced (11 features)', 'Ensemble', '0.123', '0.096', '0.509', '0.001'],
    ]
    
    table = Table(table_data, colWidths=[1.4*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.9*inch, 1.0*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(Paragraph(
        "LightGBM on Baseline 3 achieved the lowest RMSE (0.119) and was statistically unbiased (Mincer-Zarnowitz p=0.141). "
        "Notably, the best model relied on five simple features, rejecting unnecessary complexity.",
        styles['BriefTableCaption']
    ))

    story.append(Spacer(1, 0.1*inch))

    # --- Subsection 2.2: Feature Engineering Experiments ---
    story.append(Paragraph("2.2 Feature Engineering Experiments", styles['BriefSubSection']))
    
    exp_data = [
        ['Experiment', 'β Δ %', 'RMSE Δ %', 'Verdict'],
        ['Leverage Effect', '+37.8%', '+1.0%', 'Keep'],
        ['Volatility of Volatility', '+3.3%', '-1.9%', 'Reject'],
        ['EWMA Decay (λ=0.90, 0.97)', '0.0%', '0.0%', 'Reject'],
        ['Isotonic Calibration', '-18.5%', '-0.3%', 'Reject'],
        ['Log-Transform Target', '-22.7%', '-0.9%', 'Reject'],
    ]
    exp_table = Table(exp_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    exp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(exp_table)
    story.append(Paragraph(
        "Isolated experiments tested non-linear feature additions. Only the leverage effect—a feature capturing the "
        "asymmetric impact of negative returns on future volatility—improved calibration. This aligns with Black (1976).",
        styles['BriefTableCaption']
    ))

    story.append(Spacer(1, 0.1*inch))

    # --- Subsection 2.3: Economic Backtest ---
    story.append(Paragraph("2.3 Economic Backtest", styles['BriefSubSection']))
    
    bt_data = [
        ['Metric', 'Static (1x)', 'Dynamic', 'Δ'],
        ['Total Return', '-40.4%', '-29.4%', '+11.0 p.p.'],
        ['Max Drawdown', '77.2%', '66.0%', '-11.3 p.p.'],
        ['Realized Volatility', '15.0%', '11.5%', '-3.5 p.p.'],
        ['Sharpe Ratio', '-0.272', '-0.247', '+0.024'],
    ]
    bt_table = Table(bt_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    bt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(bt_table)
    story.append(Paragraph(
        "Dynamic sizing reduced maximum drawdown by 11.3 percentage points and improved total return by 11.0 p.p. "
        "The strategy undershot the 15% target (realized vol: 11.5%), indicating a conservative bias during calm regimes.",
        styles['BriefTableCaption']
    ))

    story.append(Spacer(1, 0.15*inch))

    # --- SECTION 3: KEY FIGURES ---
    story.append(Paragraph("3. Key Visualizations", styles['BriefSection']))
    story.append(Paragraph(
        "The following figures summarize the primary findings: cumulative returns, feature engineering experiments, "
        "model RMSE comparisons, and forecast calibration (Mincer-Zarnowitz).",
        styles['BriefBody']
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- VISUAL GRID (Full width) ---
    img_width = 3.4 * inch
    img_height = 2.3 * inch
    
    img1 = Image(FIGURE_PATHS['cumulative_returns'], width=img_width, height=img_height) if os.path.exists(FIGURE_PATHS['cumulative_returns']) else Paragraph("[Image not found]", styles['BriefBody'])
    img2 = Image(FIGURE_PATHS['experiment_results'], width=img_width, height=img_height) if os.path.exists(FIGURE_PATHS['experiment_results']) else Paragraph("[Image not found]", styles['BriefBody'])
    img3 = Image(FIGURE_PATHS['model_comparison'], width=img_width, height=img_height) if os.path.exists(FIGURE_PATHS['model_comparison']) else Paragraph("[Image not found]", styles['BriefBody'])
    img4 = Image(FIGURE_PATHS['mincer_zarnowitz'], width=img_width, height=img_height) if os.path.exists(FIGURE_PATHS['mincer_zarnowitz']) else Paragraph("[Image not found]", styles['BriefBody'])

    grid_data = [
        [img1, img2],
        [img3, img4],
    ]
    grid_table = Table(grid_data, colWidths=[img_width, img_width])
    grid_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(grid_table)

    # --- FOOTER: OPEN SOURCE & PAPER ---
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "The full codebase, data pipelines, and extended 10-page research paper are available on GitHub.",
        styles['BriefFooter']
    ))
    story.append(Paragraph(
        "https://github.com/kira-ml/volatility-position-sizing.git",
        styles['BriefFooter']
    ))

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def generate_pdf():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=LETTER,
        leftMargin=0.7*inch,
        rightMargin=0.7*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
    )
    
    styles = create_styles()
    story = []
    build_content(styles, story)
    
    doc.build(story)
    print(f"✅ Research Brief generated: {OUTPUT_PDF}")

if __name__ == '__main__':
    generate_pdf()