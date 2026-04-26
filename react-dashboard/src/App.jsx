import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  RadialBar,
  RadialBarChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import SectionCard from "./components/SectionCard";
import dashboardData from "./data/dashboardData";

function PrettyTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="pretty-tooltip">
      <p className="tooltip-title">{label}</p>
      {payload.map((item) => (
        <p key={`${label}-${item.name}`} className="tooltip-row">
          <span style={{ color: item.color }}>{item.name}</span>
          <strong>{typeof item.value === "number" ? item.value.toFixed(1) : item.value}</strong>
        </p>
      ))}
    </div>
  );
}

function KpiCard({ label, value, sub, percent, tone = "primary" }) {
  return (
    <article className={`kpi-card ${tone}`}>
      <div>
        <p className="kpi-label">{label}</p>
        <p className="kpi-value">{value}</p>
        <p className="kpi-sub">{sub}</p>
      </div>
      {typeof percent === "number" ? (
        <div className="kpi-ring" style={{ "--pct": `${Math.max(0, Math.min(100, percent))}%` }}>
          <span>{percent.toFixed(0)}%</span>
        </div>
      ) : null}
    </article>
  );
}

function ConfusionMatrix({ labels, values, theme }) {
  const flat = values.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);

  const tone = theme === "dark"
    ? { start: [34, 51, 76], end: [115, 183, 255], lightText: "#eaf3ff", darkText: "#dceaff" }
    : { start: [236, 244, 252], end: [18, 53, 91], lightText: "#ffffff", darkText: "#0e2f50" };

  const getColor = (value) => {
    const t = (value - min) / (max - min || 1);
    const start = tone.start;
    const end = tone.end;
    const r = Math.round(start[0] + t * (end[0] - start[0]));
    const g = Math.round(start[1] + t * (end[1] - start[1]));
    const b = Math.round(start[2] + t * (end[2] - start[2]));
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="matrix-wrap">
      <div className="matrix-row matrix-header" style={{ gridTemplateColumns: `180px repeat(${labels.length}, minmax(90px, 1fr))` }}>
        <div className="matrix-corner">Actual / Predicted</div>
        {labels.map((label) => (
          <div key={`h-${label}`} className="matrix-label">
            {label}
          </div>
        ))}
      </div>

      {values.map((row, i) => (
        <div className="matrix-row" key={`r-${labels[i]}`} style={{ gridTemplateColumns: `180px repeat(${labels.length}, minmax(90px, 1fr))` }}>
          <div className="matrix-label row-label">{labels[i]}</div>
          {row.map((v, j) => (
            <div
              key={`c-${i}-${j}`}
              className="matrix-cell"
              style={{
                background: getColor(v),
                color: v > 0.55 ? tone.lightText : tone.darkText
              }}
            >
              {v.toFixed(2)}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState(() => {
    const saved = window.localStorage.getItem("viva-theme");
    return saved === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("viva-theme", theme);
  }, [theme]);

  const COLORS = theme === "dark"
    ? {
        primary: "#73b7ff",
        secondary: "#4f86b5",
        accent: "#36c3b2",
        warm: "#f0b35a",
        softBlue: "#2c415b",
        text: "#e6efff"
      }
    : {
        primary: "#12355b",
        secondary: "#2f6f9f",
        accent: "#1f9d8f",
        warm: "#d88a2d",
        softBlue: "#dce9f6",
        text: "#0f2239"
      };

  const { projectTitle, overallHighlights, datasetPerformance, modelComparison, crossDatasetGeneralization, confusionMatrix } = dashboardData;
  const rainColumns = Array.from({ length: 26 }, (_, i) => i);

  const TwoLineDatasetTick = ({ x, y, payload }) => {
    const raw = String(payload.value || "");
    const [from, to] = raw.split(" -> ");

    return (
      <g transform={`translate(${x},${y})`}>
        <text textAnchor="middle" fill={COLORS.text} fontSize={11}>
          <tspan x="0" dy="12">{from || raw}</tspan>
          {to ? <tspan x="0" dy="13">to {to}</tspan> : null}
        </text>
      </g>
    );
  };

  const datasetChartData = datasetPerformance.map((d) => ({
    dataset: d.dataset.replace("CREMA-D", "CREMA"),
    Accuracy: d.accuracy,
    "F1-Score": d.f1Score * 100
  }));

  const modelRankingData = [...modelComparison]
    .sort((a, b) => b.accuracy - a.accuracy)
    .map((m, i) => ({
      rank: `#${i + 1}`,
      model: m.model.replace("Audio-Visual - Temporal (Proposed)", "Temporal (Proposed)"),
      accuracy: m.accuracy,
      fill: m.proposed ? COLORS.accent : COLORS.secondary,
      color: m.proposed ? COLORS.accent : COLORS.secondary
    }));

  const crossData = crossDatasetGeneralization.map((d) => ({
    split: d.split,
    splitDisplay: d.split
      .replace("CREMA-D to RAVDESS", "CREMA-D -> RAVDESS")
      .replace("CREMA-D to AFEW", "CREMA-D -> AFEW")
      .replace("RAVDESS to CREMA-D", "RAVDESS -> CREMA-D"),
    Accuracy: d.accuracy,
    "F1-Score": d.f1Score * 100
  }));

  return (
    <>
      <div className="tech-rain" aria-hidden="true">
        {rainColumns.map((i) => (
          <span
            key={`rain-${i}`}
            className="rain-drop"
            style={{
              "--x": `${(i / rainColumns.length) * 100}%`,
              "--delay": `${(i % 7) * 0.55}s`,
              "--duration": `${6 + (i % 5) * 1.1}s`
            }}
          />
        ))}
      </div>

      <div className="app-shell">
      <header className="hero">
        <div className="hero-pulse" aria-hidden="true" />
        <div className="hero-controls">
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            aria-label="Toggle light and dark mode"
          >
            {theme === "light" ? "Dark Mode" : "Light Mode"}
          </button>
        </div>
        <div className="hero-main">
          <img src="/logo.jpg" alt="Project logo" className="hero-logo" />
          <div>
            <h1>{projectTitle}</h1>
            <p className="hero-note">Temporal Fusion Analytics for Emotion Intelligence</p>
          </div>
        </div>
      </header>

      <section className="kpi-grid">
        <KpiCard
          label="Overall Accuracy"
          value={`${overallHighlights.overallAccuracy.toFixed(1)}%`}
          sub="High-confidence multimodal performance"
          percent={overallHighlights.overallAccuracy}
        />
        <KpiCard
          label="Overall F1 Score"
          value={overallHighlights.overallF1.toFixed(2)}
          sub="Balanced precision-recall quality"
          percent={overallHighlights.overallF1 * 100}
          tone="secondary"
        />
        <KpiCard label="Datasets Covered" value={String(overallHighlights.datasetsCovered)} sub="CREMA-D, RAVDESS, AFEW" tone="accent" />
        <KpiCard label="Best Model" value="Temporal Fusion" sub={overallHighlights.bestModel} tone="warm" />
      </section>

      <div className="grid">
        <SectionCard title="Dataset-Wise Performance" className="lifted">
          <div className="chart-box chart-large">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={datasetChartData} outerRadius={114}>
                <PolarGrid stroke={theme === "dark" ? "#315070" : "#c9ddee"} />
                <PolarAngleAxis dataKey="dataset" tick={{ fill: COLORS.text, fontSize: 12 }} />
                <PolarRadiusAxis domain={[60, 100]} tick={{ fill: COLORS.text, fontSize: 10 }} />
                <Radar name="Accuracy" dataKey="Accuracy" stroke={COLORS.primary} fill={COLORS.primary} fillOpacity={0.35} />
                <Radar name="F1-Score" dataKey="F1-Score" stroke={COLORS.accent} fill={COLORS.accent} fillOpacity={0.28} />
                <Legend wrapperStyle={{ color: COLORS.text }} />
                <Tooltip content={<PrettyTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard title="Model Comparison" className="lifted">
          <div className="chart-box chart-large">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison} layout="vertical" margin={{ top: 6, right: 30, left: 20, bottom: 6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={theme === "dark" ? "#2e4a69" : "#d8e8f4"} />
                <XAxis type="number" domain={[80, 90]} tick={{ fill: COLORS.text, fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="model"
                  width={170}
                  tick={{ fill: COLORS.text, fontSize: 11 }}
                  tickFormatter={(v) => v.replace("Audio-Visual - Temporal (Proposed)", "Temporal (Proposed)")}
                />
                <Tooltip content={<PrettyTooltip />} />
                <Bar dataKey="accuracy" radius={[0, 8, 8, 0]}>
                  {modelComparison.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.proposed ? COLORS.accent : COLORS.primary} />
                  ))}
                  <LabelList dataKey="accuracy" position="right" formatter={(v) => `${v.toFixed(1)}%`} fill={COLORS.text} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard title="Ranking" className="lifted">
          <div className="chart-box chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart
                innerRadius="18%"
                outerRadius="92%"
                data={modelRankingData}
                startAngle={180}
                endAngle={-180}
              >
                <PolarAngleAxis type="number" domain={[78, 90]} tick={false} />
                <RadialBar minAngle={15} background clockWise dataKey="accuracy">
                  {modelRankingData.map((entry, index) => (
                    <Cell key={`rank-${index}`} fill={entry.fill} />
                  ))}
                </RadialBar>
                <Legend
                  iconSize={10}
                  layout="vertical"
                  verticalAlign="middle"
                  align="right"
                  wrapperStyle={{ color: COLORS.text }}
                  formatter={(value, entry, idx) => `${modelRankingData[idx].rank} ${modelRankingData[idx].model}`}
                />
                <Tooltip formatter={(v, n, item) => [`${Number(v).toFixed(1)}%`, item.payload.model]} />
              </RadialBarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard title="Cross-Dataset Generalization" className="lifted">
          <div className="chart-box chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={crossData} margin={{ top: 10, right: 10, left: 0, bottom: 24 }}>
                <defs>
                  <linearGradient id="crossAcc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.primary} stopOpacity={0.42} />
                    <stop offset="100%" stopColor={COLORS.primary} stopOpacity={0.05} />
                  </linearGradient>
                  <linearGradient id="crossF1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.accent} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={COLORS.accent} stopOpacity={0.06} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={theme === "dark" ? "#2e4a69" : "#d8e8f4"} />
                <XAxis
                  dataKey="splitDisplay"
                  interval={0}
                  tick={<TwoLineDatasetTick />}
                  height={76}
                  tickMargin={10}
                />
                <YAxis domain={[60, 90]} tick={{ fill: COLORS.text, fontSize: 12 }} />
                <Tooltip content={<PrettyTooltip />} />
                <Legend verticalAlign="top" height={24} wrapperStyle={{ color: COLORS.text }} />
                <Area type="monotone" dataKey="Accuracy" stroke={COLORS.primary} fill="url(#crossAcc)" strokeWidth={3} />
                <Area type="monotone" dataKey="F1-Score" stroke={COLORS.accent} fill="url(#crossF1)" strokeWidth={3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard title="Confusion Matrix (Proposed Model)" className="lifted">
          <ConfusionMatrix labels={confusionMatrix.labels} values={confusionMatrix.values} theme={theme} />
        </SectionCard>
      </div>
      </div>
    </>
  );
}

export default App;
