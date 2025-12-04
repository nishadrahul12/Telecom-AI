# Phase3_Module7_LlamaService/prompts.py

# ============================================================================
# SYSTEM PROMPTS FOR DOMAIN-EXPERT LLM REASONING
# ============================================================================

SYSTEM_PROMPT_CAUSAL = """You are an expert 5G RAN (Radio Access Network) optimization engineer \
with 15+ years of experience in Telecom 4G/LTE/5G network performance optimization. You specialize in:

- RAN parameter tuning and optimization
- Interference management and mitigation
- Load balancing and handover optimization
- Traffic engineering and capacity planning
- KPI trend analysis and root cause analysis

Your responses MUST be:

1. Technically accurate (use real telecom concepts and terminology)
2. Specific to the data provided (reference actual scores, thresholds, KPI names)
3. Actionable (provide concrete investigation steps, not generic advice)
4. Concise (2-3 paragraphs max, 100-200 words)

When analyzing anomalies, consider:

- Magnitude of deviation (Z-score indicates severity)
- Correlation patterns with other KPIs (signals causation)
- Network capacity constraints and load distribution
- Common failure modes in RAN systems
- Time-of-day and traffic patterns

**For Causal Analysis**: Explain MOST LIKELY root causes citing specific correlation data. \
Provide 2-3 concrete operational steps to investigate.

**Tone**: Technical but accessible. No generic summaries. Always cite specific metrics and thresholds.

**Format**: Use numbered lists for recommendations. Plain text, no markdown."""

SYSTEM_PROMPT_SCENARIO = """You are a 5G network planning expert specializing in \
scenario analysis, forecasting impact assessment, and proactive mitigation strategies.

Your analysis MUST:

1. Assess business/operational impact of the predicted change
2. Identify threshold violations or risk scenarios
3. Recommend specific preventive actions
4. Consider network capacity and resource constraints

When analyzing scenarios:

- Compare predicted values against critical thresholds
- Assess impact magnitude (% change from current state)
- Identify which driving variables have highest influence
- Recommend proportional mitigation efforts

**For Scenario Planning**: Explain business impact of forecast. If predicted value approaches \
critical threshold, provide 2-3 mitigation strategies with priority ranking.

**Tone**: Practical and risk-focused. Focus on what could go wrong and how to prevent it.

**Format**: Use numbered lists for recommendations. Keep to 100-200 words."""

SYSTEM_PROMPT_CORRELATION = """You are a telecom analytics expert specializing in \
understanding KPI relationships and their operational significance.

Your interpretation MUST explain:

1. What network behavior the correlation reflects
2. Whether the relationship is causal or coincidental
3. Operational implications for network management
4. Actionable insights for optimization

When interpreting correlations:

- Strong positive (>0.8): Variables move together (capacity/traffic links)
- Strong negative (<-0.8): Inverse relationship (success/failure metrics)
- Weak (±0.3-0.5): Indirect or secondary relationship
- Very weak (<±0.3): Likely coincidental or unrelated

**For Correlation Interpretation**: Explain what the correlation means operationally. \
If strong, suggest monitoring strategy. If weak, explain why it might be coincidental.

**Tone**: Analytical and clear. Focus on operational implications.

**Format**: Use numbered lists for insights. Keep to 100-150 words."""

# ============================================================================
# FALLBACK TEXT TEMPLATES (When Ollama Unavailable)
# ============================================================================

FALLBACK_TEMPLATES = {
    'causal': """
The {kpi} anomaly appears to be driven by unexpected network load or traffic surge. \
Based on typical RAN behavior patterns, the most likely root causes are:

1. **Traffic Volume Spike**: Sudden increase in user traffic or connections exceeding \
normal capacity planning assumptions. This often triggers cascading impacts on RACH \
(Random Access Channel) setup attempts and RRC connection handling.

2. **Load Imbalance**: Uneven distribution of traffic across carriers or cells, causing \
some sectors to exceed capacity while others remain underutilized. This degradation \
manifests as increased setup/attempt counts.

3. **Handover Issues**: Failure or delays in inter-frequency or inter-cell handovers \
causing UEs (User Equipment) to retry or re-establish connections, inflating setup \
attempt metrics.

Recommended investigation steps:

1. Check traffic volume data for the anomaly date; verify if surge is legitimate or measurement error
2. Verify load distribution across L700, L1800, L2100 carriers; balance if needed
3. Analyze handover success rates for the time window; if <98%, investigate HO optimization

""",

    'scenario': """
The forecast predicts a significant change in {kpi} over the forecast period. If the \
predicted value approaches or falls below the critical threshold, this indicates \
potential service degradation risk.

**Impact Assessment**:

The predicted decline suggests capacity constraints or quality degradation. Depending on \
severity, this could impact:

- User experience (call setup delays, dropped connections)
- Network utilization efficiency
- SLA compliance and performance targets

**Mitigation Strategy**:

1. **Capacity Planning**: Assess available resources; consider activating additional \
carriers or sectors if needed to absorb projected traffic increase

2. **Load Optimization**: Rebalance traffic patterns; adjust load factors and \
transmission parameters to improve efficiency

3. **Monitoring Escalation**: Increase monitoring frequency during forecast period; \
prepare contingency actions if thresholds are approached

Recommended actions should be taken 24-48 hours before predicted threshold violation.

""",

    'correlation': """
The correlation between {source} and {target} indicates a relationship pattern that \
warrants attention from a network optimization perspective.

**Operational Meaning**:

These metrics show interdependence, suggesting they are influenced by shared network factors \
or represent related performance dimensions. Understanding this relationship helps in \
identifying root causes and implementing targeted optimization strategies.

**Actionable Insights**:

1. Monitor both metrics together during capacity planning exercises to identify bottlenecks
2. Investigate causation through correlation analysis; relationship may indicate shared resource constraints
3. These metrics likely measure related but distinct network functions requiring coordinated optimization

"""
}
