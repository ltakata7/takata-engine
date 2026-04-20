"""Portuguese / English translation tables for the portfolio PDF.

Canonical strings live in English in `construction.py`. This module provides
a `translate()` lookup used by the PDF renderer to swap to Portuguese.
If a string isn't in the map it falls back to the original English.
"""

from __future__ import annotations

from typing import Dict


# ── Per-ETF rationale translations ────────────────────────────────────────
# Key = exact English rationale string from construction.py (AllocationSlice).
# If you add a new rationale string, add the PT equivalent here too.
RATIONALE_PT: Dict[str, str] = {
    # Vanguard
    "Broad US market, 0.03% ER": "Mercado americano amplo, taxa de administração 0,03%",
    "Developed markets diversification": "Diversificação em mercados desenvolvidos",
    "EM growth exposure": "Exposição ao crescimento de mercados emergentes",
    "Core fixed income": "Renda fixa core",
    "Inflation protection": "Proteção contra inflação",
    "Global fixed income": "Renda fixa global",
    "Real asset income": "Renda de ativos reais",
    "Tail risk hedge": "Hedge de cauda (eventos extremos)",
    # All-Weather
    "Equity growth engine — performs in growth/inflation": "Motor de crescimento — desempenho em ciclos de crescimento/inflação",
    "Global equity diversification": "Diversificação global em ações",
    "Deflation/recession protection — anti-correlated to equity": "Proteção contra deflação/recessão — anticorrelacionado a ações",
    "Moderate duration — balanced rate sensitivity": "Duração moderada — sensibilidade balanceada a juros",
    "Inflation hedge — uncorrelated to stocks and bonds": "Hedge de inflação — descorrelacionado de ações e títulos",
    "Inflation/growth surprise protection": "Proteção contra surpresas de inflação/crescimento",
    # Risk Parity
    "Low-vol equity — matches risk parity framework": "Ações de baixa volatilidade — alinha com a estrutura de paridade de risco",
    "International low-vol equity": "Ações internacionais de baixa volatilidade",
    "Emerging markets low-vol": "Baixa volatilidade em mercados emergentes",
    "Broad fixed income — high risk parity weight": "Renda fixa ampla — peso elevado na paridade de risco",
    "Duration exposure for deflation protection": "Exposição a duração para proteção contra deflação",
    "Real return protection": "Proteção de retorno real",
    "Uncorrelated risk diversifier": "Diversificador de risco descorrelacionado",
    "Real asset risk contribution": "Contribuição de risco via ativos reais",
    # Income
    "Quality dividend growth — 3.5% yield, low turnover": "Crescimento de dividendos de qualidade — yield 3,5%, baixa rotatividade",
    "Broad high-yield equity — 3% yield": "Ações de alto dividendo — yield 3%",
    "International income — 4%+ yield": "Renda internacional — yield 4%+",
    "Investment grade corporates — 5% yield": "Corporativos investment grade — yield 5%",
    "High yield credit — 7%+ yield, higher risk": "Crédito high yield — yield 7%+, risco elevado",
    "Capital preservation, low duration risk": "Preservação de capital, baixo risco de duração",
    "Monthly dividend REIT — 5.5% yield": "REIT com dividendos mensais — yield 5,5%",
    "Diversified REIT exposure — 4% yield": "Exposição diversificada a REITs — yield 4%",
    "Preferred shares — 6%+ yield, bond-like": "Ações preferenciais — yield 6%+, perfil de renda fixa",
    # Factor
    "Value factor — low P/E, high book value stocks": "Fator valor — P/L baixo, alto valor contábil",
    "Momentum factor — winners keep winning": "Fator momentum — vencedores continuam vencendo",
    "Quality factor — high ROE, stable earnings": "Fator qualidade — ROE alto, lucros estáveis",
    "Low-vol factor — defensive equity": "Fator baixa volatilidade — ações defensivas",
    "Small-cap value — highest historical premium": "Valor em small caps — maior prêmio histórico",
    "International quality tilt": "Inclinação a qualidade internacional",
    "Core bond ballast": "Lastro de renda fixa core",
    "Macro hedge": "Hedge macro",
    "Trend-following — uncorrelated alpha source": "Seguidor de tendências — fonte descorrelacionada de alfa",
    # User-override fallback
    "User-selected replacement — original rationale does not apply": "Substituição escolhida pelo usuário — justificativa original não se aplica",
}


# ── Rebalancing rules translations ────────────────────────────────────────
REBALANCING_PT: Dict[str, str] = {
    # Vanguard
    "Rebalance when any allocation drifts >5% from target":
        "Rebalancear quando qualquer alocação desviar mais de 5% do alvo",
    "Calendar rebalance: review quarterly, execute if needed":
        "Rebalanceamento de calendário: revisar trimestralmente, executar se necessário",
    "Tax-loss harvest during rebalancing when possible":
        "Aproveitar perdas fiscais (tax-loss harvesting) durante o rebalanceamento quando possível",
    "Use new contributions to rebalance before selling":
        "Usar novos aportes para rebalancear antes de vender posições",
    # All-Weather
    "Rebalance quarterly to maintain target risk parity":
        "Rebalancear trimestralmente para manter a paridade de risco alvo",
    "Rebalance after any 10%+ drawdown in a single asset class":
        "Rebalancear após qualquer queda de 10%+ em uma classe de ativo isolada",
    "Do NOT rebalance during volatility spikes — wait for VIX < 25":
        "NÃO rebalancear durante picos de volatilidade — aguardar VIX abaixo de 25",
    "Annual strategic review of macro regime assumptions":
        "Revisão estratégica anual das premissas de regime macro",
    # Risk Parity
    "Monthly risk contribution check — rebalance if any asset > 30% of total risk":
        "Checagem mensal de contribuição de risco — rebalancear se algum ativo representar mais de 30% do risco total",
    "Quarterly full rebalance to inverse-volatility weights":
        "Rebalanceamento trimestral completo para pesos inversos à volatilidade",
    "Adjust weights if 60-day realized vol changes >50% from assumption":
        "Ajustar pesos se a volatilidade realizada de 60 dias variar mais de 50% da premissa",
    "Annual review of correlation assumptions":
        "Revisão anual das premissas de correlação",
    # Income
    "Reinvest dividends to maintain target weights (DRIP)":
        "Reinvestir dividendos para manter os pesos alvo (DRIP)",
    "Rebalance semi-annually — income portfolios need less frequent adjustment":
        "Rebalancear semestralmente — portfólios de renda exigem ajustes menos frequentes",
    "Review credit quality quarterly — reduce HY if spreads widen >200bp":
        "Revisar a qualidade de crédito trimestralmente — reduzir high yield se spreads se abrirem mais de 200 bps",
    "Tax-manage: harvest losses on bond positions near year-end":
        "Gestão fiscal: aproveitar perdas em posições de renda fixa próximo ao fim do ano",
    # Factor
    "Quarterly factor momentum review — rotate underperforming factors":
        "Revisão trimestral do momentum dos fatores — rotacionar fatores de baixo desempenho",
    "Rebalance when any factor tilt drifts >3% from target":
        "Rebalancear quando qualquer inclinação de fator desviar mais de 3% do alvo",
    "Annual factor premium review — confirm value/momentum/quality still positive":
        "Revisão anual dos prêmios de fatores — confirmar que valor/momentum/qualidade seguem positivos",
    "Monitor factor crowding via AUM flows into factor ETFs":
        "Monitorar saturação de fatores via fluxos de AUM em ETFs de fator",
}


# ── Tax notes translations ────────────────────────────────────────────────
TAX_PT: Dict[str, str] = {
    "Hold bonds and REITs in tax-advantaged accounts (IRA/401k)":
        "Manter títulos de renda fixa e REITs em contas com benefício fiscal (IRA/401k)",
    "Hold equity ETFs in taxable accounts (qualified dividends, lower turnover)":
        "Manter ETFs de ações em contas tributáveis (dividendos qualificados, menor rotatividade)",
    "Harvest losses on individual positions to offset gains":
        "Aproveitar perdas em posições individuais para compensar ganhos",
    "Consider tax-lot selection (specific identification) for optimal harvesting":
        "Considerar seleção de lotes tributários (identificação específica) para colheita ótima de perdas",
}


# ── Style descriptions translations ───────────────────────────────────────
STYLE_PT: Dict[str, str] = {
    "vanguard": "Vanguard Passiva — Baixo custo, diversificação global, ajustada por idade",
    "all_weather": "All-Weather (Bridgewater) — Desempenho em qualquer regime econômico",
    "risk_parity": "Paridade de Risco — Contribuição de risco igual entre classes de ativos",
    "income": "Renda / Dividendos — Maximizar yield e fluxo de caixa",
    "factor": "Baseada em Fatores (AQR) — Inclinações para Valor + Momentum + Qualidade",
}


# ── UI / PDF static strings ───────────────────────────────────────────────
# Shared labels used by the PDF generator; keyed by English canonical.
UI_PT: Dict[str, str] = {
    # Section headers
    "PORTFOLIO CONSTRUCTION": "CONSTRUÇÃO DE PORTFÓLIO",
    "RISK PROFILE COMPARISON MATRIX": "MATRIZ COMPARATIVA DE PERFIS DE RISCO",
    "ALLOCATION SPECTRUM": "ESPECTRO DE ALOCAÇÃO",
    "ASSET ALLOCATION": "ALOCAÇÃO DE ATIVOS",
    "ALLOCATION RATIONALE": "JUSTIFICATIVA DA ALOCAÇÃO",
    "PORTFOLIO RATIONALE": "JUSTIFICATIVA DO PORTFÓLIO",
    "DCA DEPLOYMENT PLAN": "PLANO DE APORTES PROGRAMADOS (DCA)",
    "REBALANCING PROTOCOL": "PROTOCOLO DE REBALANCEAMENTO",
    "TAX NOTES": "OBSERVAÇÕES FISCAIS",
    "TABLE OF CONTENTS": "SUMÁRIO",
    # Table columns
    "CATEGORY": "CATEGORIA",
    "ROLE": "FUNÇÃO",
    "ETF": "ETF",
    "NAME": "NOME",
    "TARGET": "ALVO",
    "AMOUNT": "VALOR",
    "PROFILE": "PERFIL",
    "EQUITY": "AÇÕES",
    "BONDS": "RENDA FIXA",
    "ALTS": "ALTERNATIVOS",
    "EXPECTED RETURN": "RETORNO ESPERADO",
    # Role values
    "core": "core",
    "satellite": "satélite",
    # Cover page
    "INVESTOR AGE": "IDADE DO INVESTIDOR",
    "INVESTMENT": "INVESTIMENTO",
    "DATE": "DATA",
    "All Risk Profiles": "Todos os Perfis de Risco",
    "All Styles × All Risk Profiles": "Todos os Estilos × Todos os Perfis de Risco",
    # Risk profiles
    "AGGRESSIVE": "AGRESSIVO",
    "GROWTH": "CRESCIMENTO",
    "MODERATE": "MODERADO",
    "CONSERVATIVE": "CONSERVADOR",
    "VERY CONSERVATIVE": "MUITO CONSERVADOR",
    # Legend
    "Equity": "Ações",
    "Bonds": "Renda Fixa",
    "REITs": "REITs",
    "Gold": "Ouro",
    "Other": "Outros",
    # Footer / misc
    "CONFIDENTIAL": "CONFIDENCIAL",
    "PAGE": "PÁGINA",
    "annualized": "anualizado",
    # Horizon words used by contextual explanation
    "long horizon": "horizonte longo",
    "medium horizon": "horizonte médio",
    "short horizon": "horizonte curto",
}


def translate(text: str, lang: str = "en") -> str:
    """Translate a canonical English string to the requested language.

    Falls back to the input string if no translation is registered.
    """
    if lang != "pt":
        return text
    # Merged lookup across all categories — duplicates use the first hit
    for table in (RATIONALE_PT, REBALANCING_PT, TAX_PT, STYLE_PT, UI_PT):
        if text in table:
            return table[text]
    return text


def translate_list(items: list[str], lang: str = "en") -> list[str]:
    """Translate a list of strings in bulk."""
    return [translate(s, lang) for s in items]


# ── Contextual explanation generator ──────────────────────────────────────
# Produces the "why THIS allocation at THIS age/risk/style" paragraph for
# the PDF. Computed at construction time, returned alongside the portfolio.

_HORIZON_EN = {
    "long": "a long investment horizon (20+ years)",
    "medium": "a medium investment horizon (10-20 years)",
    "short": "a short investment horizon (under 10 years)",
}
_HORIZON_PT = {
    "long": "um horizonte de investimento longo (20+ anos)",
    "medium": "um horizonte de investimento médio (10-20 anos)",
    "short": "um horizonte de investimento curto (menos de 10 anos)",
}

_STYLE_REASON_EN = {
    "vanguard":     "This Vanguard-style allocation emphasizes low-cost broad-market index funds and minimizes turnover to capture global risk premia with minimal friction.",
    "all_weather":  "The All-Weather allocation spreads risk across four macro regimes (growth up/down, inflation up/down) so drawdowns in any one environment are cushioned by gains elsewhere.",
    "risk_parity":  "Risk Parity equalizes risk contribution across asset classes rather than dollar weights — so bonds carry higher dollar allocation since they are lower-volatility than equities.",
    "income":       "The Income allocation is sized to deliver consistent cash flow from dividends, coupons, and REIT distributions while preserving capital.",
    "factor":       "The Factor portfolio overweights empirically persistent equity premia (value, momentum, quality, low-vol) expected to outperform broad beta over long horizons.",
}
_STYLE_REASON_PT = {
    "vanguard":     "Esta alocação estilo Vanguard enfatiza fundos de índice de baixo custo e ampla cobertura, minimizando rotatividade para capturar prêmios globais de risco com baixa fricção.",
    "all_weather":  "A alocação All-Weather distribui o risco entre quatro regimes macro (crescimento alto/baixo, inflação alta/baixa), de forma que quedas em um ambiente sejam amortecidas por ganhos em outros.",
    "risk_parity":  "A Paridade de Risco equaliza a contribuição de risco entre as classes de ativos em vez dos pesos em dólar — assim a renda fixa recebe alocação maior em dólares por ter menor volatilidade que ações.",
    "income":       "A alocação de Renda é dimensionada para gerar fluxo de caixa consistente via dividendos, cupons e distribuições de REITs, ao mesmo tempo preservando capital.",
    "factor":       "O portfólio de Fatores sobrepondera prêmios empiricamente persistentes em ações (valor, momentum, qualidade, baixa volatilidade) com expectativa de superar o beta amplo em horizontes longos.",
}

_RISK_REASON_EN = {
    "aggressive":        "With an aggressive profile, equity is maximized to capture long-run growth; short-term drawdowns are accepted in exchange for higher expected compounding.",
    "growth":            "The growth profile tilts toward equities while keeping a meaningful fixed-income anchor — balancing compounding with drawdown control.",
    "moderate":          "A moderate profile holds balanced equity/bond weights so the portfolio participates in growth while cushioning volatility.",
    "conservative":      "The conservative profile emphasizes fixed income and real assets to preserve capital with modest equity participation.",
    "very_conservative": "A very conservative profile prioritizes capital preservation — the minority equity sleeve is there only to offset long-run inflation drag.",
}
_RISK_REASON_PT = {
    "aggressive":        "Com perfil agressivo, a parcela de ações é maximizada para capturar o crescimento de longo prazo; quedas de curto prazo são aceitas em troca de maior composição esperada.",
    "growth":            "O perfil de crescimento inclina-se para ações mantendo uma âncora relevante em renda fixa — equilibrando composição e controle de drawdown.",
    "moderate":          "O perfil moderado mantém pesos equilibrados de ações e renda fixa, participando do crescimento e ao mesmo tempo amortecendo a volatilidade.",
    "conservative":      "O perfil conservador enfatiza renda fixa e ativos reais para preservação de capital, com participação modesta em ações.",
    "very_conservative": "O perfil muito conservador prioriza a preservação do capital — a minoria em ações serve apenas para compensar o desgaste de longo prazo causado pela inflação.",
}


def age_to_horizon(age: int) -> str:
    """Classify age into investment horizon bucket."""
    if age < 45:
        return "long"
    if age < 60:
        return "medium"
    return "short"


def generate_contextual_explanation(age: int, risk: str, style: str, lang: str = "en") -> str:
    """Build a 2-3 sentence paragraph explaining the portfolio choice.

    Shown in the PDF's "Portfolio Rationale" section to contextualize why
    the specific style/risk mix was chosen for this investor.
    """
    horizon = age_to_horizon(age)

    if lang == "pt":
        horizon_phrase = _HORIZON_PT[horizon]
        style_reason = _STYLE_REASON_PT.get(style, "")
        risk_reason = _RISK_REASON_PT.get(risk, "")
        lead = f"O investidor tem {age} anos, o que implica {horizon_phrase}."
    else:
        horizon_phrase = _HORIZON_EN[horizon]
        style_reason = _STYLE_REASON_EN.get(style, "")
        risk_reason = _RISK_REASON_EN.get(risk, "")
        lead = f"The investor is {age} years old, implying {horizon_phrase}."

    return " ".join([lead, risk_reason, style_reason]).strip()
