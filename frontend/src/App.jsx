import React, { useState, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import {
  ShieldAlert, Shield, Server, Cpu, Database,
  Activity, CheckCircle, RefreshCw, AlertTriangle, Layers, Clock,
  ExternalLink, CheckCircle2, XCircle, HardDrive, Search, Filter,
  TrendingUp, Compass, FileText, Lock, Network, ArrowRight, Download, Zap, Radio
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from "recharts";

const API_BASE = "http://127.0.0.1:8000/api";
const SOCKET_URL = "http://127.0.0.1:8000";


function WhyThisForecastPanel({ explanation, hostIp, selectedHost, selectedHostForecast, getStageBadgeStyle }) {
  const exp = explanation || selectedHostForecast?.explanation;

  if (!exp) {
    return (
      <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs mt-5">
        <div className="flex flex-wrap justify-between items-center border-b border-base-border pb-3 mb-3 gap-2">
          <div className="flex items-center gap-2">
            <Compass className="h-4 w-4 text-accent" />
            <h3 className="font-semibold uppercase text-xs tracking-wider text-text-primary">
              WHY THIS FORECAST? <span className="text-text-muted font-normal">/ EVIDENCE-GROUNDED ATTRIBUTION</span>
            </h3>
          </div>
          <span className="text-text-muted text-[11px]">Target: <strong className="text-text-primary">{hostIp}</strong></span>
        </div>
        <div className="py-8 text-center text-text-muted space-y-1.5">
          <Shield className="h-6 w-6 mx-auto text-severity-normal mb-1 opacity-80" />
          <p className="font-semibold text-text-primary text-xs">NO ACTIVE THREAT FORECAST DETECTED</p>
          <p className="text-[11px] text-text-muted max-w-lg mx-auto">
            Host <span className="text-text-secondary">{hostIp}</span> is operating nominally in baseline monitoring state. Multi-tier evidence attribution activates upon anomalous temporal sequence forecasts.
          </p>
        </div>
      </div>
    );
  }

  const threatProb = exp?.forecast?.threat_probability ?? (selectedHostForecast?.threat_probability ?? (selectedHostForecast?.threatLevel ?? 0.0));
  const stage = exp?.forecast?.predicted_stage || selectedHostForecast?.predictedStage || selectedHost?.predictedStage || "Normal";
  const horizonSec = exp?.forecast?.forecast_horizon_seconds || 25;

  const modelEvidence = exp?.model_evidence;
  const telemetryEvidence = exp?.telemetry_evidence || [];
  const sensorEvidence = exp?.sensor_evidence;
  const fastflow = sensorEvidence?.fastflow;
  const snort = sensorEvidence?.snort;
  const stageTransition = exp?.stage_transition;
  const attackPath = exp?.attack_path;
  const uncertainty = exp?.uncertainty;
  const provenance = exp?.provenance;
  const narrative = exp?.narrative_summary;

  return (
    <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs mt-5">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center border-b border-base-border pb-3 mb-4 gap-2">
        <div className="flex items-center gap-2">
          <Compass className="h-4 w-4 text-accent" />
          <h3 className="font-semibold uppercase text-xs tracking-wider text-text-primary">
            WHY THIS FORECAST? <span className="text-text-muted font-normal">/ EVIDENCE-GROUNDED ATTRIBUTION</span>
          </h3>
        </div>
        <div className="flex items-center gap-2.5 text-[11px]">
          <span className="text-text-muted">Target: <strong className="text-text-primary">{hostIp}</strong></span>
          <span className="text-text-muted">Horizon: <strong className="text-accent">+{horizonSec}s</strong></span>
          <span className={`px-2 py-0.5 rounded border text-[10px] font-bold ${getStageBadgeStyle(stage)}`}>
            {stage}
          </span>
          <span className={`px-2 py-0.5 rounded border text-[10px] font-bold ${threatProb >= 0.7 ? "bg-severity-critical/10 text-severity-critical border-severity-critical/20" : "bg-severity-normal/10 text-severity-normal border-severity-normal/20"}`}>
            Threat Prob: {Math.round(threatProb * 100)}%
          </span>
        </div>
      </div>

      {/* Deterministic Factual Narrative Banner */}
      {narrative && (
        <div className="bg-base-bg/80 border-l-2 border-accent p-3 rounded-r-lg mb-4 text-[11px] leading-relaxed text-text-secondary">
          <span className="font-bold text-accent uppercase text-[10px] block mb-1">Synthesized Evidence Narrative</span>
          <p>{narrative}</p>
        </div>
      )}

      {/* 3-Tier Evidence Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        {/* TIER 1: MODEL EVIDENCE */}
        <div className="bg-base-bg/50 border border-base-border rounded-lg p-3 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center border-b border-base-border/50 pb-2 mb-2.5">
              <span className="text-[10px] uppercase font-bold text-accent flex items-center gap-1">
                <Cpu className="h-3 w-3" /> 1. Model Evidence
              </span>
              <span className="text-[9px] text-text-muted bg-base-surface px-1.5 py-0.5 rounded border border-base-border">
                Gradient Saliency
              </span>
            </div>
            <p className="text-[10px] text-text-muted mb-2">
              Target: <strong className="text-text-primary">{modelEvidence?.target || "attack_probability"}</strong>
            </p>
            <div className="space-y-1.5">
              {(modelEvidence?.top_temporal_features || modelEvidence?.top_features || []).slice(0, 4).map((f, i) => (
                <div key={i} className="flex justify-between items-center text-[10px] bg-base-surface/70 px-2 py-1 rounded border border-base-border/40">
                  <span className="text-text-primary font-mono truncate max-w-[150px]">
                    {f.label || f.feature}
                  </span>
                  <span className={`flex items-center gap-1 font-bold ${f.direction === "increases_risk" ? "text-severity-critical" : (f.direction === "decreases_risk" ? "text-severity-normal" : "text-text-muted")}`}>
                    {f.direction === "increases_risk" ? "↑ risk" : (f.direction === "decreases_risk" ? "↓ safe" : "—")}
                    <span className="text-[9px] text-text-muted font-normal">({(f.abs_importance !== undefined ? f.abs_importance : f.attribution)?.toFixed(3)})</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-2.5 pt-2 border-t border-base-border/40 text-[9px] text-text-muted italic">
            {modelEvidence?.summary || "Attributions computed across all 10 temporal sequence windows."}
          </div>
        </div>

        {/* TIER 2: NETWORK TELEMETRY EVIDENCE */}
        <div className="bg-base-bg/50 border border-base-border rounded-lg p-3 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center border-b border-base-border/50 pb-2 mb-2.5">
              <span className="text-[10px] uppercase font-bold text-accent flex items-center gap-1">
                <Activity className="h-3 w-3" /> 2. Network Telemetry
              </span>
              <span className="text-[9px] text-text-muted bg-base-surface px-1.5 py-0.5 rounded border border-base-border">
                23-D Physical State
              </span>
            </div>
            <p className="text-[10px] text-text-muted mb-2">
              Deviation vs recent history baseline:
            </p>
            <div className="space-y-1.5">
              {telemetryEvidence.slice(0, 4).map((t, i) => (
                <div key={i} className="flex justify-between items-center text-[10px] bg-base-surface/70 px-2 py-1 rounded border border-base-border/40">
                  <span className="text-text-primary truncate max-w-[130px]">{t.feature_label || t.feature}</span>
                  <span className={`font-bold ${t.direction === "elevated" ? "text-severity-critical" : (t.direction === "suppressed" ? "text-accent" : "text-severity-normal")}`}>
                    {t.deviation_pct >= 0 ? `+${t.deviation_pct}%` : `${t.deviation_pct}%`}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-2.5 pt-2 border-t border-base-border/40 text-[9px] text-text-muted">
            Source: <strong className="text-text-secondary">NetworkState (Flow Telemetry)</strong>
          </div>
        </div>

        {/* TIER 3: SECURITY SENSORS */}
        <div className="bg-base-bg/50 border border-base-border rounded-lg p-3 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center border-b border-base-border/50 pb-2 mb-2.5">
              <span className="text-[10px] uppercase font-bold text-accent flex items-center gap-1">
                <Shield className="h-3 w-3" /> 3. Security Sensors
              </span>
              <span className="text-[9px] text-text-muted bg-base-surface px-1.5 py-0.5 rounded border border-base-border">
                Snort & FastFlow
              </span>
            </div>
            <div className="space-y-2">
              {/* FastFlow */}
              <div className="bg-base-surface/70 p-2 rounded border border-base-border/40 text-[10px]">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="font-bold text-text-primary">FastFlowDetector:</span>
                  <span className={`px-1.5 py-0.2 rounded text-[9px] font-semibold ${fastflow?.suspicious ? "bg-severity-critical/10 text-severity-critical" : "bg-severity-normal/10 text-severity-normal"}`}>
                    {fastflow?.available ? (fastflow?.suspicious ? "ANOMALY FLAGGED" : "NOMINAL") : "INACTIVE"}
                  </span>
                </div>
                <div className="text-text-muted text-[9px]">
                  {fastflow?.available ? `Suspicion score: ${fastflow?.score?.toFixed(2)} (${fastflow?.label})` : "Per-flow model checkpoint not configured."}
                </div>
              </div>

              {/* Snort */}
              <div className="bg-base-surface/70 p-2 rounded border border-base-border/40 text-[10px]">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="font-bold text-text-primary">Snort Signature:</span>
                  <span className={`px-1.5 py-0.2 rounded text-[9px] font-semibold ${snort?.available ? "bg-severity-high/10 text-severity-high" : "bg-base-bg text-text-muted"}`}>
                    {snort?.available ? `SID ${snort.sid}` : "NONE CORRELATED"}
                  </span>
                </div>
                <div className="text-text-muted text-[9px] truncate">
                  {snort?.available ? `${snort.message} (${snort.protocol})` : "No Snort signature alert temporally matched."}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-2.5 pt-2 border-t border-base-border/40 text-[9px] text-text-muted">
            Strict timestamp-gated sensor correlation.
          </div>
        </div>
      </div>

      {/* Lower Row: Stage Transition, Attack Path, Uncertainty, Provenance */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 pt-1">
        {/* Stage Transition */}
        <div className="bg-base-bg/40 border border-base-border rounded p-2.5 text-[10px]">
          <span className="text-[9px] uppercase font-bold text-text-muted block mb-1">Stage Transition</span>
          {stageTransition?.transition_detected ? (
            <div className="flex items-center gap-1.5 my-1 font-bold text-text-primary">
              <span className="text-text-muted">{stageTransition.previous_stage}</span>
              <ArrowRight className="h-3 w-3 text-accent" />
              <span className="text-severity-high">{stageTransition.predicted_stage}</span>
            </div>
          ) : (
            <div className="text-text-secondary my-1 font-semibold">
              {stageTransition?.summary || "Nominal state; no transition."}
            </div>
          )}
          <span className="text-[9px] text-text-muted block mt-1">MITRE ATT&CK progression</span>
        </div>

        {/* Attack Path */}
        <div className="bg-base-bg/40 border border-base-border rounded p-2.5 text-[10px]">
          <span className="text-[9px] uppercase font-bold text-text-muted block mb-1">Attack Path Reconstruction</span>
          {attackPath?.available ? (
            <div className="my-1">
              <span className="font-bold text-severity-high block truncate">{attackPath.source_host} → {attackPath.destination_host}</span>
              <span className="text-[9px] text-text-muted">{attackPath.hop_count} sequential hops ({attackPath.severity})</span>
            </div>
          ) : (
            <div className="text-text-muted my-1 italic">
              No active attack path correlated.
            </div>
          )}
          <span className="text-[9px] text-text-muted block mt-1">Multi-hop lateral tracking</span>
        </div>

        {/* Uncertainty */}
        <div className="bg-base-bg/40 border border-base-border rounded p-2.5 text-[10px]">
          <span className="text-[9px] uppercase font-bold text-text-muted block mb-1">MC-Dropout Predictive Uncertainty</span>
          {uncertainty?.available ? (
            <div className="my-1 space-y-0.5">
              <div className="flex justify-between">
                <span className="text-text-muted">Empirical Std:</span>
                <span className="font-bold text-text-primary">σ = {uncertainty.std?.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Empirical Spread:</span>
                <span className="text-accent font-bold">[{Math.round((uncertainty.confidence_band?.[0] || 0)*100)}%, {Math.round((uncertainty.confidence_band?.[1] || 0)*100)}%]</span>
              </div>
            </div>
          ) : (
            <div className="text-text-muted my-1 italic">
              Predictive uncertainty unavailable
            </div>
          )}
          <span className="text-[8px] text-text-muted block mt-1">10 stochastic forward passes</span>
        </div>

        {/* Audit Provenance */}
        <div className="bg-base-bg/40 border border-base-border rounded p-2.5 text-[10px]">
          <span className="text-[9px] uppercase font-bold text-text-muted block mb-1">Audit Provenance</span>
          <div className="my-1 space-y-0.5">
            <div className="flex items-center gap-1 text-[9px] font-bold">
              {provenance?.blockchain_registered ? (
                <span className="text-severity-normal flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" /> Ledger Registered
                </span>
              ) : (
                <span className="text-text-muted flex items-center gap-1">
                  <Clock className="h-3 w-3" /> Unregistered (Nominal)
                </span>
              )}
            </div>
            <div className="text-[9px] text-text-muted truncate font-mono">
              Hash: {provenance?.data_hash ? `${provenance.data_hash.slice(0, 14)}...` : "SHA-256 Validated"}
            </div>
          </div>
          <span className="text-[8px] text-text-muted block mt-0.5 italic">Tamper-evident log; 0.0s lead-time baseline</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard"); // dashboard, forecast, traffic, mitigations, topology, alerts, audit
  const [hosts, setHosts] = useState([]);
  const [selectedHostIp, setSelectedHostIp] = useState("192.168.1.10");
  const [alerts, setAlerts] = useState([]);
  const [trafficEvents, setTrafficEvents] = useState([]);
  const [rolloutData, setRolloutData] = useState(null);
  const [loadingRollout, setLoadingRollout] = useState(false);
  const [verifyingAlert, setVerifyingAlert] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [loadingVerification, setLoadingVerification] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [latestForecast, setLatestForecast] = useState({});
  const [lastForecastTime, setLastForecastTime] = useState({});
  const [worldModelStatus, setWorldModelStatus] = useState(null);
  const [mitigations, setMitigations] = useState([]);
  const [attackPaths, setAttackPaths] = useState([]);
  const [flowDetectorStatus, setFlowDetectorStatus] = useState(null);
  const [snortStatus, setSnortStatus] = useState(null);
  const [blockchainStatus, setBlockchainStatus] = useState(null);
  const [canonicalBenchmark, setCanonicalBenchmark] = useState(null);
  const [selectedAlertForDrawer, setSelectedAlertForDrawer] = useState(null);

  const [hostExplanation, setHostExplanation] = useState(null);

  const selectedHost = hosts.find(h => h.ip === selectedHostIp) || (hosts.length > 0 ? hosts[0] : null);
  const selectedHostForecast = (selectedHost ? latestForecast[selectedHost.ip] : null) || (selectedHost ? alerts.find(a => a.hostIp === selectedHost.ip) : null);

  // Freshness calculation helpers
  const formatDataAge = (ts) => {
    if (!ts) return "Awaiting Live Data";
    const diffSec = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
    if (isNaN(diffSec) || diffSec < 0) return "Just now";
    if (diffSec < 5) return "Fresh (<5s)";
    if (diffSec < 60) return `${diffSec}s ago`;
    return `${Math.floor(diffSec / 60)}m ago`;
  };

  const isDataStale = (ts) => {
    if (!ts) return false;
    const diffSec = (Date.now() - new Date(ts).getTime()) / 1000;
    return diffSec > 20;
  };

  // Host switching & explainability retrieval (clears stale state immediately)
  useEffect(() => {
    setHostExplanation(null);
    if (selectedHostForecast?.explanation) {
      setHostExplanation(selectedHostForecast.explanation);
    } else if (selectedHostIp) {
      fetch(`${API_BASE}/forecasts/explain/${selectedHostIp}`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data) setHostExplanation(data);
        })
        .catch(() => setHostExplanation(null));
    }
  }, [selectedHostIp, selectedHostForecast]);

  // Collector UI State
  const [collectorInterfaces, setCollectorInterfaces] = useState([]);
  const [selectedInterface, setSelectedInterface] = useState("");
  const [collectorStatus, setCollectorStatus] = useState({ running: false, packets_captured: 0, flows_generated: 0, bytes_captured: 0 });
  const [loadingCollector, setLoadingCollector] = useState(false);

  // Traffic Filters
  const [trafficSearch, setTrafficSearch] = useState("");
  const [protocolFilter, setProtocolFilter] = useState("ALL");
  const [alertSearch, setAlertSearch] = useState("");
  const [alertSeverityFilter, setAlertSeverityFilter] = useState("ALL");

  const socketRef = useRef(null);

  // Load initial data
  useEffect(() => {
    fetchHosts();
    fetchAlerts();
    fetchAttackPaths();
    fetchCollectorInterfaces();
    fetchCollectorStatus();
    fetchWorldModelStatus();
    fetchMitigations();
    fetchFlowDetectorStatus();
    fetchSnortStatus();
    fetchBlockchainStatus();
    fetchCanonicalBenchmark();

    // Setup Socket.io Connection
    socketRef.current = io(SOCKET_URL, {
      transports: ["polling", "websocket"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000
    });

    socketRef.current.on("connect", () => {
      setIsConnected(true);
      console.log("Connected to backend Socket.io server");
    });

    socketRef.current.on("disconnect", () => {
      setIsConnected(false);
    });

    socketRef.current.on("traffic_update", (event) => {
      console.log("[FRONTEND_FLOW]", event);
      setTrafficEvents(prev => [event, ...prev].slice(0, 300));
    });

    socketRef.current.on("collector_status", (status) => {
      setCollectorStatus(status);
      if (status && status.running && status.interface) {
        setSelectedInterface(status.interface);
      }
    });

    socketRef.current.on("forecast_update", (forecast) => {
      setLatestForecast(prev => ({
        ...prev,
        [forecast.hostIp]: forecast
      }));
      setLastForecastTime(prev => ({
        ...prev,
        [forecast.hostIp]: Date.now()
      }));
    });

    socketRef.current.on("forecast_alert", (alert) => {
      setAlerts(prev => {
        if (prev.some(a => a._id === alert._id)) return prev;
        return [alert, ...prev].slice(0, 100);
      });
      console.warn("Forecast Alert Triggered:", alert);
    });

    socketRef.current.on("host_status_change", ({ ip, status }) => {
      setHosts(prev => prev.map(h => h.ip === ip ? { ...h, status } : h));
      fetchMitigations();
    });

    socketRef.current.on("attack_path_update", (paths) => {
      setAttackPaths(paths);
    });

    // Background poll for collector status every 3s as fallback
    const pollTimer = setInterval(() => {
      fetchCollectorStatus();
      fetchWorldModelStatus();
      fetchMitigations();
      fetchFlowDetectorStatus();
      fetchSnortStatus();
      fetchAttackPaths();
      fetchBlockchainStatus();
    }, 3000);

    return () => {
      clearInterval(pollTimer);
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  const fetchAttackPaths = async () => {
    try {
      const res = await fetch(`${API_BASE}/topology/attack_paths`);
      const data = await res.json();
      setAttackPaths(data);
    } catch (err) {
      console.error("Error fetching attack paths:", err);
    }
  };

  const fetchHosts = async () => {
    try {
      const res = await fetch(`${API_BASE}/hosts`);
      const data = await res.json();
      setHosts(data);
      if (data.length > 0 && !selectedHostIp) {
        setSelectedHostIp(data[0].ip);
      }
    } catch (err) {
      console.error("Error fetching hosts:", err);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/alerts`);
      const data = await res.json();
      setAlerts(data);
    } catch (err) {
      console.error("Error fetching alerts:", err);
    }
  };

  const fetchMitigations = async () => {
    try {
      const res = await fetch(`${API_BASE}/mitigations`);
      if (res.ok) {
        const data = await res.json();
        setMitigations(data);
      }
    } catch (err) {
      console.error("Error fetching mitigations:", err);
    }
  };

  const fetchFlowDetectorStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/flow_detector/status`);
      if (res.ok) {
        const data = await res.json();
        setFlowDetectorStatus(data);
      }
    } catch (err) {
      console.error("Error fetching flow detector status:", err);
    }
  };

  const fetchSnortStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/snort/status`);
      if (res.ok) {
        const data = await res.json();
        setSnortStatus(data);
      }
    } catch (err) {
      console.error("Error fetching snort status:", err);
    }
  };

  const fetchBlockchainStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/blockchain/status`);
      if (res.ok) {
        const data = await res.json();
        setBlockchainStatus(data);
      }
    } catch (err) {
      console.error("Error fetching blockchain status:", err);
    }
  };

  const fetchCanonicalBenchmark = async () => {
    try {
      const res = await fetch(`${API_BASE}/benchmark/canonical`);
      if (res.ok) {
        const data = await res.json();
        setCanonicalBenchmark(data);
      }
    } catch (err) {
      console.error("Error fetching canonical benchmark:", err);
    }
  };

  const exportFlowsToCSV = () => {
    if (trafficEvents.length === 0) return;
    const headers = ["timestamp", "hostIp", "dstIp", "protocol", "duration", "total_bytes", "action"];
    const rows = trafficEvents.map(e => [
      e.timestamp || "",
      e.hostIp || "",
      e.dstIp || "",
      e.protocol || "",
      e.duration || "",
      e.total_bytes || "",
      e.action || ""
    ]);
    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `cyberforecaster_flows_${new Date().toISOString().slice(0, 19)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getDynamicSubnet = () => {
    const activeIf = collectorInterfaces.find(i => (typeof i === 'object' && i.name === selectedInterface) || i === selectedInterface);
    if (activeIf && typeof activeIf === 'object' && activeIf.ip && activeIf.ip !== 'N/A' && !activeIf.ip.startsWith('169.254.')) {
      const parts = activeIf.ip.split(".");
      if (parts.length === 4) {
        return `Subnet: ${parts[0]}.${parts[1]}.${parts[2]}.0/24 (Capture IP: ${activeIf.ip})`;
      }
    }
    return "Dynamic Capture Subnet";
  };

  const fetchWorldModelStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/world_model/status`);
      if (res.ok) {
        const data = await res.json();
        setWorldModelStatus(data);
      }
    } catch (err) {
      console.error("Error fetching world model status:", err);
    }
  };

  const fetchCollectorInterfaces = async () => {
    try {
      const res = await fetch(`${API_BASE}/collector/interfaces`);
      const data = await res.json();
      if (data.interfaces && data.interfaces.length > 0) {
        setCollectorInterfaces(data.interfaces);
        if (!selectedInterface) {
          const activeWithIp = data.interfaces.find(i => typeof i === 'object' && i.ip && i.ip !== 'N/A' && !i.ip.startsWith('169.254.'));
          const defaultIf = activeWithIp ? activeWithIp.name : (typeof data.interfaces[0] === 'string' ? data.interfaces[0] : (data.interfaces[0].name || "eth0"));
          setSelectedInterface(defaultIf);
        }
      }
    } catch (err) {
      console.error("Error fetching collector interfaces:", err);
    }
  };

  const fetchCollectorStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/collector/status`);
      const data = await res.json();
      setCollectorStatus(data);
      if (data && data.running && data.interface) {
        setSelectedInterface(data.interface);
      }
    } catch (err) {
      console.error("Error fetching collector status:", err);
    }
  };

  const startCollector = async () => {
    setLoadingCollector(true);
    try {
      const res = await fetch(`${API_BASE}/collector/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ interface: selectedInterface })
      });
      if (res.ok) {
        fetchCollectorStatus();
      }
    } catch (err) {
      console.error("Error starting collector:", err);
    } finally {
      setLoadingCollector(false);
    }
  };

  const stopCollector = async () => {
    setLoadingCollector(true);
    try {
      const res = await fetch(`${API_BASE}/collector/stop`, { method: "POST" });
      if (res.ok) {
        fetchCollectorStatus();
      }
    } catch (err) {
      console.error("Error stopping collector:", err);
    } finally {
      setLoadingCollector(false);
    }
  };

  // Load rollout simulation for selected host
  useEffect(() => {
    if (!selectedHostIp) return;
    loadRollout(selectedHostIp);
  }, [selectedHostIp]);

  const loadRollout = async (ip) => {
    setLoadingRollout(true);
    try {
      const res = await fetch(`${API_BASE}/forecasts/rollout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hostIp: ip })
      });
      if (res.ok) {
        const data = await res.json();
        setRolloutData(data);
      } else {
        setRolloutData(null);
      }
    } catch (err) {
      console.error("Error loading rollout:", err);
      setRolloutData(null);
    } finally {
      setLoadingRollout(false);
    }
  };

  const handleDefensiveAction = async (ip, action) => {
    try {
      const res = await fetch(`${API_BASE}/hosts/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ip, action })
      });
      if (res.ok) {
        fetchHosts();
        setTimeout(() => loadRollout(ip), 800);
      }
    } catch (err) {
      console.error("Error sending defensive action:", err);
    }
  };

  const verifyBlockchain = async (alert) => {
    setVerifyingAlert(alert);
    setLoadingVerification(true);
    setVerificationResult(null);
    try {
      const res = await fetch(`${API_BASE}/blockchain/verify/${alert._id}`);
      const data = await res.json();
      setVerificationResult(data);
    } catch (err) {
      console.error("Error verifying blockchain:", err);
      setVerificationResult({ error: "Failed to connect to local Hardhat node or contract" });
    } finally {
      setLoadingVerification(false);
    }
  };

  // Mitigation Chart formatting
  const getChartData = () => {
    if (!rolloutData || !rolloutData.scenarios) return [];
    const stepsCount = rolloutData.rollout_steps || 5;
    const chartData = [];

    for (let t = 0; t < stepsCount; t++) {
      chartData.push({
        step: `+${(t + 1) * 5}s`,
        "Do Nothing (Risk)": Math.round((rolloutData.scenarios.do_nothing?.[t]?.threat_level || 0) * 100),
        "Rate Limit": Math.round((rolloutData.scenarios.rate_limit?.[t]?.threat_level || 0) * 100),
        "Block Ports": Math.round((rolloutData.scenarios.block_port?.[t]?.threat_level || 0) * 100),
        "Isolate Host": Math.round((rolloutData.scenarios.isolate_host?.[t]?.threat_level || 0) * 100)
      });
    }
    return chartData;
  };

  // Network Activity Chart formatting (recent flow sizes/rates)
  const getLiveActivityChartData = () => {
    if (trafficEvents.length === 0) return [];
    return trafficEvents.slice(0, 20).reverse().map((ev, idx) => ({
      index: `#${idx + 1}`,
      bytes: ev.total_bytes || 64,
      duration: Math.round((ev.duration || 0.01) * 1000),
      ip: ev.hostIp
    }));
  };

// (selectedHost and selectedHostForecast declared at top of component)
  const currentWarmup = selectedHostForecast?.warmupStatus || {
    windowsCollected: worldModelStatus?.windows_collected || 0,
    windowsRequired: 10,
    isReady: (worldModelStatus?.windows_collected || 0) >= 10,
    status: (worldModelStatus?.windows_collected || 0) >= 10 ? "READY" : "WARMING UP"
  };

  const getStageBadgeStyle = (stage) => {
    switch (stage) {
      case "Data Exfiltration": return "bg-severity-critical/10 border-severity-critical/20 text-severity-critical";
      case "Command & Control": return "bg-severity-critical/10 border-severity-critical/20 text-severity-critical";
      case "Lateral Movement": return "bg-severity-high/10 border-severity-high/20 text-severity-high";
      case "Initial Access": return "bg-severity-medium/10 border-severity-medium/20 text-severity-medium";
      case "Reconnaissance": return "bg-accent/10 border-accent/20 text-accent";
      default: return "bg-severity-normal/10 border-severity-normal/20 text-severity-normal";
    }
  };

  // Filtered Traffic Events
  const filteredTraffic = trafficEvents.filter(ev => {
    const matchesSearch = trafficSearch === "" ||
      ev.hostIp?.toLowerCase().includes(trafficSearch.toLowerCase()) ||
      ev.dstIp?.toLowerCase().includes(trafficSearch.toLowerCase());

    const protoStr = (typeof ev.protocol === "string" ? ev.protocol : (ev.protocol === 1 ? "TCP" : (ev.protocol === 0.5 ? "UDP" : "ICMP"))).toUpperCase();
    const matchesProto = protocolFilter === "ALL" || protoStr === protocolFilter;
    return matchesSearch && matchesProto;
  });

  // Filtered Alerts
  const filteredAlerts = alerts.filter(a => {
    const matchesSearch = alertSearch === "" ||
      a.hostIp?.toLowerCase().includes(alertSearch.toLowerCase()) ||
      a.predictedStage?.toLowerCase().includes(alertSearch.toLowerCase());

    const matchesSeverity = alertSeverityFilter === "ALL" || a.severity === alertSeverityFilter;
    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="min-h-screen dashboard-grid pb-12 font-sans select-none text-text-primary">
      {/* Header bar */}
      <header className="border-b border-base-border bg-base-surface/90 px-6 py-3 flex justify-between items-center sticky top-0 z-40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-md bg-accent/10 border border-accent/20 text-accent">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-wider text-text-primary font-mono-tech">
              CYBERFORECASTER <span className="text-text-muted font-normal">/ TEMPORAL WORLD MODEL SOC</span>
            </h1>
            <p className="text-[11px] text-text-muted font-mono-tech">
              AI-BASED ATTACK FORECASTING · NTRO TACTICAL OPERATIONS
            </p>
          </div>
        </div>

        {/* SOC Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-base-bg p-1 rounded-lg border border-base-border text-xs font-mono-tech">
          {[
            { id: "dashboard", label: "Dashboard", icon: Layers },
            { id: "forecast", label: "Forecast", icon: TrendingUp },
            { id: "traffic", label: "Live Traffic", icon: Activity },
            { id: "mitigations", label: "Mitigations", icon: Shield },
            { id: "topology", label: "Topology", icon: Network },
            { id: "alerts", label: "Alerts", icon: ShieldAlert },
            { id: "audit", label: "Audit Log", icon: Database }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeTab === tab.id
                  ? "bg-accent text-white font-semibold shadow-sm"
                  : "text-text-secondary hover:text-text-primary hover:bg-base-surface"
              }`}
            >
              <tab.icon className="h-3.5 w-3.5" />
              <span>{tab.label}</span>
              {tab.id === "alerts" && alerts.length > 0 && (
                <span className="ml-1 px-1.5 py-0.2 text-[9px] rounded-full bg-severity-critical text-white">
                  {alerts.length}
                </span>
              )}
              {tab.id === "mitigations" && hosts.some(h => h.status !== "ONLINE") && (
                <span className="ml-1 px-1.5 py-0.2 text-[9px] rounded-full bg-severity-high text-white">
                  {hosts.filter(h => h.status !== "ONLINE").length}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Indicators */}
        <div className="flex items-center gap-2.5 text-xs font-mono-tech">
          <div className="flex items-center gap-1.5 bg-base-bg border border-base-border px-2.5 py-1 rounded text-text-muted text-[10px]">
            <Zap className="h-3 w-3 text-accent" />
            <span>FLOW DETECTOR: <strong className={flowDetectorStatus?.status === 'ACTIVE' ? 'text-severity-normal' : 'text-text-secondary'}>{flowDetectorStatus?.status || 'STANDBY'}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 bg-base-bg border border-base-border px-2.5 py-1 rounded text-text-muted text-[10px]">
            <Radio className="h-3 w-3 text-accent" />
            <span>SNORT: <strong className={snortStatus?.connected ? 'text-severity-normal' : 'text-text-secondary'}>{snortStatus?.status || 'NOT CONNECTED'}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 bg-base-bg border border-base-border px-2.5 py-1 rounded text-text-muted text-[10px]">
            <Database className="h-3 w-3 text-accent" />
            <span>LEDGER: <strong className={blockchainStatus?.connected ? 'text-severity-normal' : 'text-text-muted'}>{blockchainStatus?.connected ? 'ONLINE' : 'OFFLINE'}</strong></span>
          </div>

          <div className="flex items-center gap-2 bg-base-bg border border-severity-normal/20 px-2.5 py-1 rounded text-severity-normal text-[10px]">
            <div className={`h-1.5 w-1.5 rounded-full ${currentWarmup.isReady ? "bg-severity-normal pulse-indicator" : "bg-accent"}`}></div>
            <span>WORLD MODEL {currentWarmup.status}</span>
          </div>

          <div className={`flex items-center gap-2 bg-base-bg px-2.5 py-1 rounded border text-[10px] ${isConnected ? "border-base-border text-text-secondary" : "border-severity-critical/30 text-severity-critical"}`}>
            <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? "bg-severity-normal" : "bg-severity-critical"}`}></div>
            <span>{isConnected ? "ONLINE" : "DISCONNECTED"}</span>
          </div>
        </div>
      </header>

      {/* Top Metrics Ribbon */}
      <section className="grid grid-cols-5 gap-4 px-6 mt-5">
        {[
          {
            title: "Protected Assets",
            value: hosts.length > 0 ? hosts.length : 5,
            sub: "Monitored endpoints",
            icon: Server,
            valueColor: "text-text-primary"
          },
          {
            title: "Active High-Threat Alerts",
            value: alerts.filter(a => a.severity === "HIGH" || a.severity === "CRITICAL").length,
            sub: "Genuine threat signals",
            icon: ShieldAlert,
            valueColor: alerts.some(a => a.severity === "HIGH" || a.severity === "CRITICAL") ? "text-severity-critical" : "text-text-primary"
          },
          {
            title: "Flows Processed",
            value: collectorStatus.flows_generated.toLocaleString(),
            sub: `${collectorStatus.packets_captured.toLocaleString()} raw packets`,
            icon: Activity,
            valueColor: "text-accent"
          },
          {
            title: "Recent Flow Buffer",
            value: `${trafficEvents.length} / 300`,
            sub: "Live inspection window",
            icon: Clock,
            valueColor: "text-text-secondary"
          },
          {
            title: "On-Chain Predictions",
            value: alerts.filter(a => a.blockchainTxHash).length,
            sub: "Tamper-proof ledger",
            icon: Database,
            valueColor: "text-severity-normal"
          }
        ].map((item, idx) => (
          <div key={idx} className="bg-base-surface border border-base-border rounded-lg p-4 flex items-center justify-between card-panel-hover">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">{item.title}</p>
              <h2 className={`text-xl font-bold mt-0.5 ${item.valueColor} font-mono-tech`}>{item.value}</h2>
              <p className="text-[10px] text-text-muted font-mono-tech mt-0.5">{item.sub}</p>
            </div>
            <item.icon className="h-6 w-6 text-text-muted/60" />
          </div>
        ))}
      </section>

      {/* Main Tab Content */}
      <div className="px-6 mt-5">

        {/* ----------------- 1. DASHBOARD TAB ----------------- */}
        {activeTab === "dashboard" && (
          <div className="grid grid-cols-12 gap-5">
            {/* Left Column: World Model Forecast Overview & Top Flow Stream */}
            <div className="col-span-8 flex flex-col gap-5">
              {/* World Model Status & Live Rollout Card */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">
                      Temporal World Model Forecaster
                    </h3>
                  </div>
                  <div className="flex items-center gap-2.5 text-xs font-mono-tech">
                    <span className="text-text-muted">Target: <strong className="text-text-primary">{selectedHost?.ip || "Awaiting Assets"}</strong></span>
                    {selectedHostForecast?.timestamp && (
                      <span className={`px-2 py-0.5 rounded border text-[10px] ${isDataStale(selectedHostForecast.timestamp) ? "bg-severity-medium/10 text-severity-medium border-severity-medium/30" : "bg-base-bg text-text-secondary border-base-border"}`}>
                        {isDataStale(selectedHostForecast.timestamp) ? `STALE (${formatDataAge(selectedHostForecast.timestamp)})` : formatDataAge(selectedHostForecast.timestamp)}
                      </span>
                    )}
                    <span className={`px-2 py-0.5 rounded border text-[10px] ${getStageBadgeStyle(selectedHostForecast?.predictedStage || selectedHost?.predictedStage || "Normal")}`}>
                      {selectedHostForecast?.predictedStage || selectedHost?.predictedStage || "Normal"}
                    </span>
                  </div>
                </div>

                {/* Warm-Up Status Bar */}
                <div className="bg-base-bg/70 border border-base-border rounded-lg p-3 mb-4">
                  <div className="flex justify-between items-center text-xs font-mono-tech mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-text-secondary font-semibold">Temporal Context Buffer:</span>
                      <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${currentWarmup.isReady ? "bg-severity-normal/10 text-severity-normal" : "bg-accent/10 text-accent"}`}>
                        {currentWarmup.status} ({currentWarmup.windowsCollected} / {currentWarmup.windowsRequired} Windows)
                      </span>
                    </div>
                    <span className="text-text-muted text-[10px]">5.0s Network Windows</span>
                  </div>
                  <div className="w-full bg-base-surface rounded-full h-2 overflow-hidden border border-base-border">
                    <div
                      className={`h-full transition-all duration-500 ${currentWarmup.isReady ? "bg-severity-normal" : "bg-accent"}`}
                      style={{ width: `${Math.min(100, (currentWarmup.windowsCollected / currentWarmup.windowsRequired) * 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* 5-Step Future Horizon Trajectory */}
                <div className="border border-base-border rounded-lg p-3.5 bg-base-bg/40">
                  <div className="flex justify-between items-baseline mb-3">
                    <h4 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">
                      Autoregressive Multi-Step Horizon Trajectory (+5s to +25s)
                    </h4>
                    <span className="text-[10px] text-text-muted font-mono-tech">Recursive Temporal LSTM Rollout</span>
                  </div>

                  <div className="grid grid-cols-6 gap-2 text-center font-mono-tech">
                    {/* Step 0: Current */}
                    <div className="bg-base-surface border border-accent/30 rounded p-2.5">
                      <span className="text-[9px] text-text-muted block">CURRENT</span>
                      <span className="text-xs font-bold text-text-primary mt-1 block">T+0s</span>
                      <span className="text-[10px] text-accent mt-1 block font-semibold truncate">
                        {selectedHostForecast?.predictedStage || selectedHost?.predictedStage || "Normal"}
                      </span>
                      <span className="text-[9px] text-text-muted block mt-0.5">
                        Threat: {selectedHostForecast ? `${Math.round((selectedHostForecast.threat_probability ?? selectedHostForecast.threatLevel ?? 0) * 100)}%` : "0% (Nominal)"}
                      </span>
                    </div>

                    {/* Step 1 to 5 from rollout */}
                    {[1, 2, 3, 4, 5].map((stepIdx) => {
                      const stepData = selectedHostForecast?.rollout?.[stepIdx - 1];
                      const hasStep = !!stepData;
                      const stepRisk = hasStep ? Math.round(stepData.attack_probability * 100) : null;
                      const stepStage = hasStep ? (stepData.predicted_stage_id === 0 ? "Normal" : `Stage ${stepData.predicted_stage_id}`) : null;

                      return (
                        <div key={stepIdx} className="bg-base-surface border border-base-border rounded p-2.5">
                          <span className="text-[9px] text-text-muted block">HORIZON</span>
                          <span className="text-xs font-bold text-text-secondary mt-1 block">+{stepIdx * 5}s</span>
                          {hasStep ? (
                            <>
                              <span className={`text-[10px] mt-1 block font-semibold truncate ${stepRisk > 50 ? "text-severity-critical" : "text-severity-normal"}`}>
                                {stepStage}
                              </span>
                              <span className="text-[9px] text-text-muted block mt-0.5">
                                Threat: {stepRisk}%
                              </span>
                            </>
                          ) : (
                            <>
                              <span className="text-[10px] mt-1 block text-text-muted italic">
                                Awaiting
                              </span>
                              <span className="text-[9px] text-text-muted block mt-0.5">
                                —
                              </span>
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Evidence-Grounded Forecast Explainability Panel */}
                <WhyThisForecastPanel
                  explanation={hostExplanation}
                  hostIp={selectedHost?.ip || "192.168.1.10"}
                  selectedHost={selectedHost}
                  selectedHostForecast={selectedHostForecast}
                  getStageBadgeStyle={getStageBadgeStyle}
                />
              </div>

              {/* Live Traffic Recent Preview Table */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3.5">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">
                      Live Flow Activity Feed
                    </h3>
                  </div>
                  <button
                    onClick={() => setActiveTab("traffic")}
                    className="text-[10px] text-accent hover:underline font-mono-tech flex items-center gap-1"
                  >
                    <span>View All {trafficEvents.length} Buffered Flows</span>
                    <ArrowRight className="h-3 w-3" />
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono-tech text-xs">
                    <thead>
                      <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px]">
                        <th className="py-2 font-semibold">Time</th>
                        <th className="font-semibold">Source</th>
                        <th className="font-semibold">Destination</th>
                        <th className="font-semibold">Proto</th>
                        <th className="font-semibold">Duration</th>
                        <th className="text-right font-semibold">Volume</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-base-border/50">
                      {trafficEvents.length === 0 ? (
                        <tr>
                          <td colSpan="6" className="py-6 text-center text-text-muted">
                            Awaiting network packet capture. Click START CAPTURE on the Live Traffic panel.
                          </td>
                        </tr>
                      ) : (
                        trafficEvents.slice(0, 6).map((ev, idx) => (
                          <tr key={idx} className="hover:bg-base-surfaceHover/50 transition-colors">
                            <td className="py-2 text-text-muted text-[10px]">
                              {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "Live"}
                            </td>
                            <td className="font-semibold text-text-primary">{ev.hostIp}</td>
                            <td className="text-text-secondary">{ev.dstIp}</td>
                            <td>
                              <span className="px-1.5 py-0.2 rounded text-[9px] bg-base-bg border border-base-border text-text-secondary">
                                {ev.protocol === 1 ? "TCP" : (ev.protocol === 0.5 ? "UDP" : "ICMP")}
                              </span>
                            </td>
                            <td className="text-text-muted">{ev.duration?.toFixed(3)}s</td>
                            <td className="text-right text-accent font-semibold">{ev.total_bytes?.toLocaleString()} B</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Right Column: Monitored Assets & Collector Control */}
            <div className="col-span-4 flex flex-col gap-5">
              {/* Live Capture Control Panel */}
              <div className="bg-base-surface border border-base-border rounded-lg p-4 font-mono-tech text-xs">
                <div className="flex justify-between items-center border-b border-base-border pb-2.5 mb-3">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                      Packet Collector
                    </h3>
                  </div>
                  <span className={`text-[10px] px-2 py-0.5 rounded border ${collectorStatus.running ? "bg-severity-normal/10 border-severity-normal/20 text-severity-normal" : "bg-base-bg border-base-border text-text-muted"}`}>
                    {collectorStatus.running ? "CAPTURING LIVE" : "IDLE"}
                  </span>
                </div>

                <div className="space-y-2.5">
                  <div>
                    <label className="text-[10px] text-text-muted block mb-1">NETWORK ADAPTER</label>
                    <select
                      value={selectedInterface}
                      onChange={(e) => setSelectedInterface(e.target.value)}
                      className="w-full bg-base-bg border border-base-border rounded px-2.5 py-1.5 text-text-primary focus:outline-none focus:border-accent text-xs font-mono-tech truncate"
                    >
                      {collectorInterfaces.length === 0 ? (
                        <option value="eth0">eth0 (Default)</option>
                      ) : (
                        collectorInterfaces.map((iface, idx) => {
                          const val = typeof iface === 'string' ? iface : (iface.name || `iface-${idx}`);
                          const displayName = typeof iface === 'object' && iface.display_name ? iface.display_name : (typeof iface === 'string' ? iface : (iface.name || 'Unknown'));
                          const ipStr = typeof iface === 'object' && iface.ip && iface.ip !== 'N/A' ? ` (${iface.ip})` : '';
                          const desc = typeof iface === 'object' && iface.description ? ` - ${iface.description}` : '';
                          return <option key={idx} value={val}>{displayName}{ipStr}{desc}</option>;
                        })
                      )}
                    </select>
                  </div>

                  <div className="flex gap-2 pt-1">
                    {!collectorStatus.running ? (
                      <button
                        onClick={startCollector}
                        disabled={loadingCollector}
                        className="w-full py-2 rounded bg-severity-normal/10 border border-severity-normal/30 hover:bg-severity-normal/20 text-severity-normal font-semibold transition-colors disabled:opacity-40"
                      >
                        {loadingCollector ? "Starting..." : "START CAPTURE"}
                      </button>
                    ) : (
                      <button
                        onClick={stopCollector}
                        disabled={loadingCollector}
                        className="w-full py-2 rounded bg-severity-critical/10 border border-severity-critical/30 hover:bg-severity-critical/20 text-severity-critical font-semibold transition-colors disabled:opacity-40"
                      >
                        {loadingCollector ? "Stopping..." : "STOP CAPTURE"}
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Monitored Assets List */}
              <div className="bg-base-surface border border-base-border rounded-lg p-4 flex flex-col flex-1">
                <div className="flex justify-between items-center border-b border-base-border pb-2.5 mb-3">
                  <div className="flex items-center gap-2">
                    <Server className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">
                      Monitored Assets
                    </h3>
                  </div>
                  <span className="text-[10px] text-text-muted font-mono-tech">{hosts.length} Protected</span>
                </div>

                <div className="space-y-2 overflow-y-auto max-h-[300px] pr-1">
                  {hosts.map(h => (
                    <div
                      key={h.ip}
                      onClick={() => setSelectedHostIp(h.ip)}
                      className={`p-3 rounded-lg border transition-all cursor-pointer font-mono-tech ${
                        h.ip === selectedHostIp
                          ? "bg-accent/10 border-accent text-text-primary ring-1 ring-accent"
                          : "bg-base-bg border-base-border hover:border-base-borderActive text-text-secondary"
                      }`}
                    >
                      <div className="flex justify-between items-center text-xs">
                        <span className="font-bold text-text-primary">{h.name}</span>
                        <span className={`text-[10px] px-1.5 py-0.2 rounded border ${getStageBadgeStyle(h.predictedStage)}`}>
                          {h.predictedStage}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-[10px] text-text-muted mt-1">
                        <span>{h.ip}</span>
                        <span>{h.department}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ----------------- 2. FORECAST TAB (HIGH PRIORITY) ----------------- */}
        {activeTab === "forecast" && (
          <div className="grid grid-cols-12 gap-5">
            {/* World Model Horizon & Predicted State Details */}
            <div className="col-span-8 flex flex-col gap-5">
              {/* World Model Forecaster Card */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-accent" />
                    <div>
                      <h3 className="font-bold uppercase text-sm tracking-wider font-mono-tech text-text-primary">
                        World Model Multi-Step Forecast
                      </h3>
                      <p className="text-[10px] text-text-muted font-mono-tech">
                        Autoregressive Neural State Rollout (+5s to +25s Horizons)
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 text-xs font-mono-tech">
                    <span className="text-text-muted">Target Asset:</span>
                    <select
                      value={selectedHostIp}
                      onChange={(e) => setSelectedHostIp(e.target.value)}
                      className="bg-base-bg border border-base-border rounded px-2 py-1 text-text-primary focus:outline-none focus:border-accent text-xs font-mono-tech"
                    >
                      {hosts.map(h => (
                        <option key={h.ip} value={h.ip}>{h.name} ({h.ip})</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Warm-Up Banner */}
                <div className="bg-base-bg border border-base-border rounded-lg p-4 mb-4">
                  <div className="flex justify-between items-center text-xs font-mono-tech mb-2">
                    <span className="font-semibold text-text-primary">
                      World Model Readiness State:
                    </span>
                    <span className={`px-2 py-0.5 rounded font-bold border ${currentWarmup.isReady ? "bg-severity-normal/10 border-severity-normal/20 text-severity-normal" : "bg-accent/10 border-accent/20 text-accent"}`}>
                      {currentWarmup.status} ({currentWarmup.windowsCollected} / {currentWarmup.windowsRequired} Windows)
                    </span>
                  </div>
                  <div className="w-full bg-base-surface rounded-full h-2.5 overflow-hidden border border-base-border">
                    <div
                      className={`h-full transition-all duration-500 ${currentWarmup.isReady ? "bg-severity-normal" : "bg-accent"}`}
                      style={{ width: `${Math.min(100, (currentWarmup.windowsCollected / currentWarmup.windowsRequired) * 100)}%` }}
                    ></div>
                  </div>
                  <p className="text-[10px] text-text-muted font-mono-tech mt-2">
                    Requires a continuous history of 10 sequential 5-second network state windows for autoregressive hidden-state initialization.
                  </p>
                </div>

                {/* Horizon Rollout Cards */}
                <div className="space-y-3">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">
                    Multi-Horizon State Trajectory
                  </h4>

                  <div className="grid grid-cols-5 gap-3 font-mono-tech text-xs">
                    {[1, 2, 3, 4, 5].map((stepIdx) => {
                      const stepData = selectedHostForecast?.rollout?.[stepIdx - 1];
                      const hasStep = !!stepData;
                      const stepRisk = hasStep ? Math.round(stepData.attack_probability * 100) : null;
                      const stageName = hasStep ? (stepData.predicted_stage_id === 0 ? "Normal" : `Stage ${stepData.predicted_stage_id}`) : null;

                      return (
                        <div key={stepIdx} className="bg-base-bg border border-base-border rounded-lg p-3 flex flex-col justify-between">
                          <div>
                            <div className="flex justify-between items-center text-[10px] text-text-muted border-b border-base-border/50 pb-1 mb-2">
                              <span>HORIZON</span>
                              <span className="font-bold text-text-primary">+{stepIdx * 5}s</span>
                            </div>
                            {hasStep ? (
                              <>
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold inline-block ${getStageBadgeStyle(stageName)}`}>
                                  {stageName}
                                </span>
                                <div className="mt-2.5">
                                  <span className="text-[10px] text-text-muted block">Threat Probability:</span>
                                  <span className={`text-lg font-bold ${stepRisk > 50 ? "text-severity-critical" : "text-severity-normal"}`}>
                                    {stepRisk}%
                                  </span>
                                </div>
                              </>
                            ) : (
                              <div className="py-4 text-center text-text-muted">
                                <span className="text-[10px] block italic">Awaiting Rollout</span>
                                <span className="text-xs text-text-muted block mt-1">—</span>
                              </div>
                            )}
                          </div>

                          <div className="mt-3 pt-2 border-t border-base-border/50 text-[9px] text-text-muted space-y-0.5">
                            <div>Horizon Step: <span className="text-text-secondary">+{stepIdx * 5}s</span></div>
                            <div>Status: <span className="text-text-secondary">{hasStep ? "Projected" : "Standby"}</span></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Evidence-Grounded Forecast Explainability Panel */}
                <WhyThisForecastPanel
                  explanation={hostExplanation}
                  hostIp={selectedHost?.ip || selectedHostIp}
                  selectedHost={selectedHost}
                  selectedHostForecast={selectedHostForecast}
                  getStageBadgeStyle={getStageBadgeStyle}
                />

                {/* "What-If" Counterfactual Simulation Plot */}
                <div className="mt-5 border border-base-border p-4 rounded-lg bg-base-bg/40">
                  <div className="flex justify-between items-baseline mb-2">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">
                      "What-If" Defensive Counterfactual Trajectories
                    </h4>
                    <span className="text-[9px] text-text-muted font-mono-tech italic">
                      illustrative mitigation impact (heuristic, not model-learned)
                    </span>
                  </div>

                  <div className="h-[180px] w-full mt-2">
                    <ResponsiveContainer width="100%" height={180}>
                      <AreaChart data={getChartData()} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorDoNothing" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f0506e" stopOpacity={0.15}/>
                            <stop offset="95%" stopColor="#f0506e" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorIsolate" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3ecf8e" stopOpacity={0.15}/>
                            <stop offset="95%" stopColor="#3ecf8e" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                        <XAxis dataKey="step" stroke="#5a6275" tick={{ fontSize: 9, fill: "#8b93a7" }} />
                        <YAxis domain={[0, 100]} stroke="#5a6275" tick={{ fontSize: 9, fill: "#8b93a7" }} />
                        <Tooltip contentStyle={{ backgroundColor: "#12161f", borderColor: "#232838", borderRadius: "6px", color: "#e6e8ec", fontSize: "11px" }} />
                        <Legend wrapperStyle={{ fontSize: '9px', paddingTop: '6px' }} />
                        <Area type="monotone" dataKey="Do Nothing (Risk)" stroke="#f0506e" strokeWidth={1.5} fillOpacity={1} fill="url(#colorDoNothing)" />
                        <Area type="monotone" dataKey="Rate Limit" stroke="#f0a050" strokeWidth={1.5} fillOpacity={0} />
                        <Area type="monotone" dataKey="Block Ports" stroke="#a78bfa" strokeWidth={1.5} fillOpacity={0} />
                        <Area type="monotone" dataKey="Isolate Host" stroke="#3ecf8e" strokeWidth={1.5} fillOpacity={1} fill="url(#colorIsolate)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Model Status & Validated Benchmark Reference */}
            <div className="col-span-4 flex flex-col gap-5">
              {/* World Model Status Panel */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs">
                <div className="flex items-center gap-2 border-b border-base-border pb-3 mb-3.5">
                  <Cpu className="h-4 w-4 text-accent" />
                  <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                    Model Status & Architecture
                  </h3>
                </div>

                <div className="space-y-2.5">
                  <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                    <span className="text-text-muted">Architecture</span>
                    <span className="font-bold text-text-primary">Temporal LSTM (23-D)</span>
                  </div>
                  <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                    <span className="text-text-muted">Model Status</span>
                    <span className={`font-bold ${currentWarmup.isReady ? "text-severity-normal" : "text-accent"}`}>
                      {currentWarmup.status}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                    <span className="text-text-muted">Scaler Discipline</span>
                    <span className="text-text-secondary font-semibold">Train-Fitted StateScaler</span>
                  </div>
                  <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                    <span className="text-text-muted">Uncertainty Metric</span>
                    <span className="text-text-muted italic">Uncertainty unavailable</span>
                  </div>
                  <div className="flex flex-col border-b border-base-border/50 pb-1.5">
                    <span className="text-text-muted">Loaded Checkpoint</span>
                    <span className="text-[10px] text-text-secondary mt-0.5 truncate">
                      models_weights/lstm_world_model.pt
                    </span>
                  </div>
                </div>
              </div>

              {/* Canonical Benchmark Validation Summary */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3.5">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-severity-normal" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                      Validated Benchmark Metrics
                    </h3>
                  </div>
                  <span className="px-1.5 py-0.5 rounded text-[8.5px] bg-accent/10 border border-accent/20 text-accent font-bold">
                    OFFLINE HELD-OUT
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-baseline mb-1">
                      <span className="text-[10px] text-text-muted block">CANONICAL RMSE (K=1, +5s Horizon)</span>
                      <span className="text-[8.5px] text-text-muted">CSE-CIC-IDS2018 Test Split</span>
                    </div>
                    <div className="grid grid-cols-3 gap-1.5 mt-1 text-center text-[10px]">
                      <div className="bg-base-bg p-2 rounded border border-base-border">
                        <span className="text-text-muted block text-[8px]">PERSISTENCE</span>
                        <span className="font-bold text-severity-critical">
                          {canonicalBenchmark?.future_state_rmse_benchmarks?.horizon_K_1?.Persistence_Baseline ?? (worldModelStatus?.canonical_rmse_reference?.K_step_1?.persistence_rmse ?? "11.74")}
                        </span>
                      </div>
                      <div className="bg-base-bg p-2 rounded border border-base-border">
                        <span className="text-text-muted block text-[8px]">TRAIN MEAN</span>
                        <span className="font-bold text-text-secondary">
                          {canonicalBenchmark?.future_state_rmse_benchmarks?.horizon_K_1?.Training_Mean_Baseline ?? (worldModelStatus?.canonical_rmse_reference?.K_step_1?.training_mean_rmse ?? "2.15")}
                        </span>
                      </div>
                      <div className="bg-base-bg p-2 rounded border border-severity-normal/30 bg-severity-normal/5">
                        <span className="text-severity-normal block text-[8px]">LSTM WORLD MODEL</span>
                        <span className="font-bold text-severity-normal">
                          {canonicalBenchmark?.future_state_rmse_benchmarks?.horizon_K_1?.Temporal_LSTM_World_Model ?? (worldModelStatus?.canonical_rmse_reference?.K_step_1?.lstm_rmse ?? "2.05")}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Multi-Horizon Reference */}
                  <div className="bg-base-bg/60 p-2 rounded border border-base-border text-[9.5px] space-y-1">
                    <span className="text-[8.5px] uppercase font-bold text-text-muted block">Extended Horizons (LSTM vs Persistence):</span>
                    <div className="flex justify-between">
                      <span className="text-text-muted">+15s (K=3):</span>
                      <span>LSTM <strong className="text-severity-normal">2.35</strong> vs Persist <strong className="text-text-muted">12.00</strong></span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">+25s (K=5):</span>
                      <span>LSTM <strong className="text-severity-normal">2.16</strong> vs Persist <strong className="text-text-muted">12.01</strong></span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-base-border/50 space-y-1.5">
                    <div className="flex justify-between items-baseline">
                      <span className="text-[10px] text-text-muted">VALIDATED LEAD TIME:</span>
                      <span className="text-xs font-bold text-severity-normal">0.0s (Exact-Onset)</span>
                    </div>
                    <div className="flex justify-between text-[9px] text-text-muted">
                      <span>Pre-Onset Alarms on Baseline:</span>
                      <span className="text-text-primary font-semibold">0 (Zero False Alarms)</span>
                    </div>
                    <p className="text-[8.5px] text-text-muted leading-relaxed italic pt-0.5">
                      Static offline benchmark results on CSE-CIC-IDS2018 untouched test split; not modified by live inference telemetry.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ----------------- 3. LIVE TRAFFIC TAB ----------------- */}
        {activeTab === "traffic" && (
          <div className="space-y-5">
            {/* Live Activity Chart & Controls */}
            <div className="grid grid-cols-12 gap-5">
              <div className="col-span-8 bg-base-surface border border-base-border rounded-lg p-5">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">
                      Live Network Activity Stream (Flow Volume & Packet Durations)
                    </h3>
                  </div>
                  <span className="text-[10px] text-text-muted font-mono-tech">Real-Time Ingestion</span>
                </div>

                <div className="h-[140px] w-full">
                  <ResponsiveContainer width="100%" height={140}>
                    <AreaChart data={getLiveActivityChartData()} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                      <XAxis dataKey="index" stroke="#5a6275" tick={{ fontSize: 9, fill: "#8b93a7" }} />
                      <YAxis stroke="#5a6275" tick={{ fontSize: 9, fill: "#8b93a7" }} />
                      <Tooltip contentStyle={{ backgroundColor: "#12161f", borderColor: "#232838", borderRadius: "6px", color: "#e6e8ec", fontSize: "11px" }} />
                      <Area type="monotone" dataKey="bytes" name="Bytes Volume" stroke="#3b82f6" strokeWidth={1.5} fill="url(#colorTraffic)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Collector Stats Box */}
              <div className="col-span-4 bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 border-b border-base-border pb-2.5 mb-3">
                    <HardDrive className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                      Capture Metrics
                    </h3>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Packets Captured:</span>
                      <span className="font-bold text-text-primary">{collectorStatus.packets_captured.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Flows Processed:</span>
                      <span className="font-bold text-accent">{collectorStatus.flows_generated.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Total Volume:</span>
                      <span className="font-bold text-text-secondary">{(((collectorStatus && typeof collectorStatus.bytes_captured === 'number') ? collectorStatus.bytes_captured : 0) / 1024).toFixed(1)} KB</span>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-base-border">
                  <span className="text-[10px] text-text-muted block mb-1.5">CAPTURE INTERFACE</span>
                  <span className="text-xs font-bold text-text-primary block truncate">
                    {collectorStatus.interface || selectedInterface || "Wi-Fi"}
                  </span>
                </div>
              </div>
            </div>

            {/* Filterable Live Flow Table */}
            <div className="bg-base-surface border border-base-border rounded-lg p-5">
              <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-text-muted" />
                    <input
                      type="text"
                      placeholder="Search IP / Protocol..."
                      value={trafficSearch}
                      onChange={(e) => setTrafficSearch(e.target.value)}
                      className="bg-base-bg border border-base-border rounded pl-8 pr-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent font-mono-tech w-64"
                    />
                  </div>

                  {/* Protocol Filters */}
                  <div className="flex items-center gap-1 font-mono-tech text-xs">
                    {["ALL", "TCP", "UDP", "ICMP"].map(p => (
                      <button
                        key={p}
                        onClick={() => setProtocolFilter(p)}
                        className={`px-2.5 py-1 rounded border text-[11px] transition-all ${
                          protocolFilter === p
                            ? "bg-accent/10 border-accent text-accent font-bold"
                            : "bg-base-bg border-base-border text-text-muted hover:text-text-primary"
                        }`}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={exportFlowsToCSV}
                    disabled={trafficEvents.length === 0}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-primary text-xs font-mono-tech transition-colors disabled:opacity-40"
                  >
                    <Download className="h-3.5 w-3.5 text-accent" />
                    Export CSV ({trafficEvents.length})
                  </button>
                  <div className="text-xs font-mono-tech text-text-muted">
                    Showing <strong className="text-text-primary">{filteredTraffic.length}</strong> of {trafficEvents.length} Buffered Flows
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto max-h-[420px]">
                <table className="w-full text-left font-mono-tech text-xs">
                  <thead className="sticky top-0 bg-base-surface z-10">
                    <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px]">
                      <th className="py-2.5 font-semibold">Timestamp</th>
                      <th className="font-semibold">Source IP</th>
                      <th className="font-semibold">Destination IP</th>
                      <th className="font-semibold">Protocol</th>
                      <th className="font-semibold">Duration</th>
                      <th className="font-semibold">Total Bytes</th>
                      <th className="font-semibold">Risk Class</th>
                      <th className="text-right font-semibold">Action Tag</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-base-border/50">
                    {filteredTraffic.length === 0 ? (
                      <tr>
                        <td colSpan="8" className="py-8 text-center text-text-muted">
                          No matching network traffic flows found.
                        </td>
                      </tr>
                    ) : (
                      filteredTraffic.map((ev, idx) => (
                        <tr key={idx} className="hover:bg-base-surfaceHover/50 transition-colors">
                          <td className="py-2 text-text-muted text-[10px]">
                            {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "Live"}
                          </td>
                          <td className="font-semibold text-text-primary">{ev.hostIp}</td>
                          <td className="text-text-secondary">{ev.dstIp}</td>
                          <td>
                            <span className="px-1.5 py-0.2 rounded text-[9px] bg-base-bg border border-base-border text-text-secondary font-bold">
                              {typeof ev.protocol === 'string' ? ev.protocol : (ev.protocol === 1 ? "TCP" : (ev.protocol === 0.5 ? "UDP" : "ICMP"))}
                            </span>
                          </td>
                          <td className="text-text-muted">{ev.duration?.toFixed(3)}s</td>
                          <td className="text-accent font-semibold">{ev.total_bytes?.toLocaleString()} B</td>
                          <td>
                            {ev.fast_detection && ev.fast_detection.available ? (
                              ev.fast_detection.suspicious ? (
                                <span className="px-1.5 py-0.2 rounded text-[9px] bg-severity-critical/10 border border-severity-critical/30 text-severity-critical font-bold">
                                  BOT / SUSPICIOUS ({Math.round(ev.fast_detection.confidence * 100)}%)
                                </span>
                              ) : (
                                <span className="px-1.5 py-0.2 rounded text-[9px] bg-severity-normal/10 border border-severity-normal/20 text-severity-normal">
                                  BENIGN
                                </span>
                              )
                            ) : (
                              <span className="px-1.5 py-0.2 rounded text-[9px] bg-severity-normal/10 border border-severity-normal/20 text-severity-normal">
                                BENIGN
                              </span>
                            )}
                          </td>
                          <td className="text-right text-text-muted text-[10px]">
                            {ev.action === 1 ? (
                              <span className="text-severity-medium font-bold">[ACT: RATE LIMIT]</span>
                            ) : ev.action === 2 ? (
                              <span className="text-severity-high font-bold">[ACT: PORTS BLOCKED]</span>
                            ) : ev.action === 3 ? (
                              <span className="text-severity-critical font-bold">[ACT: ISOLATED]</span>
                            ) : (
                              "PASSED"
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ----------------- MITIGATIONS TAB ----------------- */}
        {activeTab === "mitigations" && (
          <div className="grid grid-cols-12 gap-5">
            <div className="col-span-8 bg-base-surface border border-base-border rounded-lg p-5">
              <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-accent" />
                  <div>
                    <h3 className="font-bold uppercase text-xs tracking-wider font-mono-tech text-text-primary">
                      Active Defense & Mitigation Interventions
                    </h3>
                    <p className="text-[10px] text-text-muted font-mono-tech">
                      Enforce dynamic ACLs, port blocks, flow rate-limits, and quarantine containment.
                    </p>
                  </div>
                </div>
                <span className="text-xs font-mono-tech text-text-secondary">
                  <strong>{hosts.filter(h => h.status !== 'ONLINE').length}</strong> Active Host Mitigations
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono-tech text-xs">
                  <thead>
                    <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px]">
                      <th className="py-2.5 font-semibold">Host Name</th>
                      <th className="font-semibold">IP Address</th>
                      <th className="font-semibold">Active Policy</th>
                      <th className="font-semibold">Criticality</th>
                      <th className="font-semibold">Current State</th>
                      <th className="text-right font-semibold">Defensive Controls</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-base-border/50">
                    {hosts.map(h => (
                      <tr key={h.ip} className="hover:bg-base-surfaceHover/50 transition-colors">
                        <td className="py-3 font-bold text-text-primary">{h.name}</td>
                        <td className="text-text-secondary">{h.ip}</td>
                        <td>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            h.status === 'ISOLATED' ? 'bg-severity-critical/10 text-severity-critical border border-severity-critical/30' :
                            (h.status === 'PORTS_BLOCKED' ? 'bg-severity-high/10 text-severity-high border border-severity-high/30' :
                            (h.status === 'RATE_LIMITED' ? 'bg-severity-medium/10 text-severity-medium border border-severity-medium/30' :
                            'bg-severity-normal/10 text-severity-normal border border-severity-normal/20'))
                          }`}>
                            {h.status}
                          </span>
                        </td>
                        <td className="text-accent font-semibold">{h.criticality}</td>
                        <td className="text-text-muted text-[10px]">{h.department}</td>
                        <td className="text-right py-3">
                          <div className="flex items-center justify-end gap-1.5">
                            {h.status === 'ONLINE' ? (
                              <>
                                <button
                                  onClick={() => handleDefensiveAction(h.ip, "RATE_LIMIT")}
                                  className="px-2 py-1 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-[10px] text-text-secondary transition-colors"
                                >
                                  Rate Limit
                                </button>
                                <button
                                  onClick={() => handleDefensiveAction(h.ip, "BLOCK_PORTS")}
                                  className="px-2 py-1 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-[10px] text-severity-high transition-colors"
                                >
                                  Block Ports
                                </button>
                                <button
                                  onClick={() => handleDefensiveAction(h.ip, "ISOLATE")}
                                  className="px-2 py-1 rounded bg-severity-critical/10 border border-severity-critical/30 hover:bg-severity-critical/20 text-[10px] text-severity-critical font-bold transition-colors"
                                >
                                  Isolate
                                </button>
                              </>
                            ) : (
                              <button
                                onClick={() => handleDefensiveAction(h.ip, "RESET")}
                                className="px-3 py-1 rounded bg-severity-normal/10 border border-severity-normal/30 hover:bg-severity-normal/20 text-[10px] text-severity-normal font-bold transition-colors"
                              >
                                Release / Reset to Online
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Mitigation History Log */}
            <div className="col-span-4 bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs">
              <div className="flex items-center gap-2 border-b border-base-border pb-3 mb-3">
                <Clock className="h-4 w-4 text-accent" />
                <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                  Mitigation Action Log
                </h3>
              </div>

              <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
                {mitigations.length === 0 ? (
                  <p className="text-center py-8 text-text-muted text-[11px]">
                    No mitigation events recorded. All enterprise hosts operating normally.
                  </p>
                ) : (
                  mitigations.map((m, idx) => (
                    <div key={idx} className="bg-base-bg border border-base-border rounded p-2.5">
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-bold text-text-primary text-[11px]">{m.hostIp}</span>
                        <span className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${m.active ? 'bg-severity-critical/10 text-severity-critical' : 'bg-severity-normal/10 text-severity-normal'}`}>
                          {m.action}
                        </span>
                      </div>
                      <p className="text-[10px] text-text-muted leading-relaxed">{m.reason}</p>
                      <span className="text-[9px] text-text-muted block mt-1">
                        {new Date(m.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* ----------------- 4. TOPOLOGY TAB ----------------- */}
        {activeTab === "topology" && (
          <div className="grid grid-cols-12 gap-5">
            {/* Asset Topology Canvas */}
            <div className="col-span-8 bg-base-surface border border-base-border rounded-lg p-5 flex flex-col min-h-[460px]">
              <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3.5">
                <div className="flex items-center gap-2">
                  <Network className="h-5 w-5 text-accent" />
                  <div>
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">
                      Monitored Asset Topology
                    </h3>
                    <p className="text-[10px] text-text-muted font-mono-tech">
                      {getDynamicSubnet()}
                    </p>
                  </div>
                </div>
                <span className="text-[10px] text-text-muted font-mono-tech">Real-Time Edge Highlights Active</span>
              </div>

              {/* SVG Topology Canvas */}
              <div className="flex-1 relative flex items-center justify-center bg-base-bg/60 rounded-md border border-base-border/60 overflow-hidden min-h-[340px]">
                <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
                  {hosts.map((h, idx) => {
                    const angle = (idx * 2 * Math.PI) / hosts.length;
                    const x = 50 + 32 * Math.cos(angle);
                    const y = 50 + 32 * Math.sin(angle);

                    const isHighThreat = latestForecast[h.ip]?.predictedStage === "Data Exfiltration" ||
                                         latestForecast[h.ip]?.predictedStage === "Command & Control";
                    const lineColor = isHighThreat ? "rgba(240, 80, 110, 0.6)" : "#232838";
                    const strokeWidth = isHighThreat ? "2" : "1";

                    return (
                      <line
                        key={idx}
                        x1="50%"
                        y1="50%"
                        x2={`${x}%`}
                        y2={`${y}%`}
                        stroke={lineColor}
                        strokeWidth={strokeWidth}
                        strokeDasharray={isHighThreat ? "4 4" : "0"}
                      />
                    );
                  })}
                </svg>

                {/* Central Gateway Node */}
                <div className="absolute z-10 flex flex-col items-center justify-center bg-base-surface border border-accent/40 rounded-md p-3 shadow-lg">
                  <Cpu className="h-6 w-6 text-accent" />
                  <span className="text-[10px] font-mono-tech mt-1 font-bold text-text-primary">GATEWAY</span>
                  <span className="text-[8px] font-mono-tech text-text-muted">192.168.1.1</span>
                </div>

                {/* Dynamic Host Nodes */}
                {hosts.map((h, idx) => {
                  const angle = (idx * 2 * Math.PI) / hosts.length;
                  const x = 50 + 32 * Math.cos(angle);
                  const y = 50 + 32 * Math.sin(angle);

                  const hostForecast = latestForecast[h.ip];
                  const stage = hostForecast ? hostForecast.predictedStage : "Normal";
                  const isSelected = h.ip === selectedHostIp;

                  let borderClass = "border-base-border bg-base-surface text-text-primary";
                  let iconColor = "text-text-muted";

                  if (h.status === "ISOLATED") {
                    borderClass = "border-severity-critical/40 bg-severity-critical/10 text-severity-critical opacity-70";
                    iconColor = "text-severity-critical";
                  } else if (stage === "Data Exfiltration" || stage === "Command & Control") {
                    borderClass = "border-severity-critical/60 bg-severity-critical/10 text-severity-critical";
                    iconColor = "text-severity-critical";
                  } else if (stage === "Lateral Movement" || stage === "Initial Access") {
                    borderClass = "border-severity-high/60 bg-severity-high/10 text-severity-high";
                    iconColor = "text-severity-high";
                  } else if (stage === "Reconnaissance") {
                    borderClass = "border-accent/50 bg-accent/5 text-accent";
                    iconColor = "text-accent";
                  } else if (isSelected) {
                    borderClass = "border-accent bg-accent/10 ring-1 ring-accent text-text-primary";
                    iconColor = "text-accent";
                  }

                  return (
                    <div
                      key={h.ip}
                      onClick={() => setSelectedHostIp(h.ip)}
                      style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%, -50%)' }}
                      className={`absolute z-10 flex flex-col items-center justify-center p-2.5 rounded-lg border shadow-md transition-all cursor-pointer hover:scale-105 ${borderClass}`}
                    >
                      <Server className={`h-5 w-5 ${iconColor}`} />
                      <span className="text-[10px] font-mono-tech font-bold mt-1 max-w-[85px] truncate text-center">{h.name}</span>
                      <span className="text-[8px] font-mono-tech text-text-muted">{h.ip}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Asset Control Drawer */}
            <div className="col-span-4 flex flex-col gap-5">
              {/* Selected Host Details */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3">
                  <span className="font-semibold text-text-secondary uppercase text-[11px]">Asset Telemetry</span>
                  <span className={`px-2 py-0.5 rounded border text-[10px] ${getStageBadgeStyle(selectedHost?.predictedStage || "Normal")}`}>
                    {selectedHost?.predictedStage || "Normal"}
                  </span>
                </div>

                {!selectedHost ? (
                  <div className="py-6 text-center text-text-muted">
                    No asset selected.
                  </div>
                ) : (
                  <>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-text-muted">Host Name:</span>
                        <span className="font-bold text-text-primary">{selectedHost.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">IP Address:</span>
                        <span className="font-semibold text-text-secondary">{selectedHost.ip}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Department:</span>
                        <span className="text-text-secondary">{selectedHost.department}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Criticality:</span>
                        <span className="text-accent font-semibold">{selectedHost.criticality}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Operating Status:</span>
                        <span className="font-bold text-severity-normal">{selectedHost.status}</span>
                      </div>
                    </div>

                    {/* Defensive Action Buttons */}
                    <div className="mt-4 pt-3 border-t border-base-border">
                      <span className="text-[10px] text-text-muted block mb-2 font-semibold">DEFENSIVE INTERVENTIONS</span>
                      <div className="grid grid-cols-2 gap-2 text-xs font-mono-tech">
                        <button
                          onClick={() => handleDefensiveAction(selectedHost.ip, "RATE_LIMIT")}
                          disabled={selectedHost.status === "RATE_LIMITED" || selectedHost.status === "ISOLATED"}
                          className="p-2 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-primary text-[10px] transition-colors disabled:opacity-30"
                        >
                          Rate Limit Flows
                        </button>
                        <button
                          onClick={() => handleDefensiveAction(selectedHost.ip, "BLOCK_PORTS")}
                          disabled={selectedHost.status === "PORTS_BLOCKED" || selectedHost.status === "ISOLATED"}
                          className="p-2 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-primary text-[10px] transition-colors disabled:opacity-30"
                        >
                          Block Ports
                        </button>
                        <button
                          onClick={() => handleDefensiveAction(selectedHost.ip, "ISOLATE")}
                          disabled={selectedHost.status === "ISOLATED"}
                          className="col-span-2 p-2 rounded bg-severity-critical/10 border border-severity-critical/30 hover:bg-severity-critical/20 text-severity-critical text-[10px] font-bold transition-colors disabled:opacity-30"
                        >
                          Isolate Host (Quarantine)
                        </button>
                        {selectedHost.status !== "ONLINE" && (
                          <button
                            onClick={() => handleDefensiveAction(selectedHost.ip, "RESET")}
                            className="col-span-2 p-2 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-secondary text-[10px] transition-colors"
                          >
                            Reset Host to Normal
                          </button>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Observed Attack Path Box */}
              <div className="bg-base-surface border border-base-border rounded-lg p-5 font-mono-tech text-xs">
                <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3">
                  <div className="flex items-center gap-2">
                    <Compass className="h-4 w-4 text-accent" />
                    <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary">
                      Attack Path Reconstruction
                    </h3>
                  </div>
                  <span className="px-1.5 py-0.5 rounded text-[8.5px] bg-accent/10 border border-accent/20 text-accent font-bold">
                    OBSERVED TELEMETRY
                  </span>
                </div>

                {attackPaths.length === 0 ? (
                  <div className="py-4 text-center text-text-muted text-xs">
                    <Shield className="h-6 w-6 mx-auto mb-2 text-severity-normal" />
                    <p className="font-bold text-text-primary">NO ATTACK PATH ESTABLISHED</p>
                    <p className="text-[10px] text-text-muted mt-1">
                      No multi-hop lateral progression detected in current live telemetry.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3 max-h-[240px] overflow-y-auto pr-1">
                    {attackPaths.map((path) => (
                      <div key={path.path_id} className="p-3 bg-base-bg/90 border border-severity-critical/40 rounded-md">
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-severity-critical/20 text-severity-critical border border-severity-critical/30">
                            {path.severity} LATERAL PROGRESSION
                          </span>
                          <span className="text-[9px] text-text-muted font-mono-tech">{path.mitre_technique}</span>
                        </div>
                        <div className="flex items-center gap-1.5 flex-wrap my-2 text-[11px] font-bold">
                          <span className="text-text-primary">{path.source}</span>
                          {path.hops.map((hop, hIdx) => (
                            <span key={hIdx} className="flex items-center gap-1 text-severity-critical">
                              <span>&rarr;</span>
                              <span>{hop.to}</span>
                            </span>
                          ))}
                        </div>
                        <div className="text-[9px] text-text-muted border-t border-base-border/50 pt-1.5 mt-1.5 space-y-1">
                          {path.hops.map((h, i) => (
                            <div key={i} className="flex justify-between text-[8.5px]">
                              <span>Hop {i + 1} ({h.protocol}): {h.from} &rarr; {h.to}</span>
                              <span className="text-accent truncate max-w-[140px]">{h.evidence}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ----------------- 5. ALERTS TAB ----------------- */}
        {activeTab === "alerts" && (
          <div className="bg-base-surface border border-base-border rounded-lg p-5">
            <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-text-muted" />
                  <input
                    type="text"
                    placeholder="Search Target IP / Threat..."
                    value={alertSearch}
                    onChange={(e) => setAlertSearch(e.target.value)}
                    className="bg-base-bg border border-base-border rounded pl-8 pr-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent font-mono-tech w-64"
                  />
                </div>

                <div className="flex items-center gap-1 font-mono-tech text-xs">
                  {["ALL", "CRITICAL", "HIGH", "MEDIUM"].map(s => (
                    <button
                      key={s}
                      onClick={() => setAlertSeverityFilter(s)}
                      className={`px-2.5 py-1 rounded border text-[11px] transition-all ${
                        alertSeverityFilter === s
                          ? "bg-accent/10 border-accent text-accent font-bold"
                          : "bg-base-bg border-base-border text-text-muted hover:text-text-primary"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              <div className="text-xs font-mono-tech text-text-muted">
                <strong>{filteredAlerts.length}</strong> Total Threat Records
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono-tech text-xs">
                <thead>
                  <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px]">
                    <th className="py-2.5 font-semibold">Timestamp</th>
                    <th className="font-semibold">Target IP</th>
                    <th className="font-semibold">Severity</th>
                    <th className="font-semibold">Forecasted Threat Stage</th>
                    <th className="font-semibold">Threat Probability</th>
                    <th className="font-semibold">MITRE Techniques</th>
                    <th className="font-semibold">Blockchain Audit</th>
                    <th className="text-right font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-base-border/50">
                  {filteredAlerts.length === 0 ? (
                    <tr>
                      <td colSpan="8" className="py-10 text-center text-text-muted">
                        No active threat forecasting alerts recorded. Network remains secure.
                      </td>
                    </tr>
                  ) : (
                    filteredAlerts.map(alert => (
                      <tr
                        key={alert._id}
                        onClick={() => setSelectedAlertForDrawer(alert)}
                        className="hover:bg-base-surfaceHover/50 transition-colors cursor-pointer"
                      >
                        <td className="py-2.5 text-text-muted text-[10px]">
                          {new Date(alert.timestamp).toLocaleString()}
                        </td>
                        <td className="font-semibold text-text-primary">{alert.hostIp}</td>
                        <td>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${alert.severity === "CRITICAL" ? "bg-severity-critical/10 text-severity-critical border border-severity-critical/20" : "bg-severity-high/10 text-severity-high border border-severity-high/20"}`}>
                            {alert.severity}
                          </span>
                        </td>
                        <td>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-medium border ${getStageBadgeStyle(alert.predictedStage)}`}>
                            {alert.predictedStage}
                          </span>
                        </td>
                        <td className="text-text-secondary">{(((alert.threat_probability !== undefined ? alert.threat_probability : (alert.attack_probability !== undefined ? alert.attack_probability : alert.confidence)) || 0) * 100).toFixed(1)}%</td>
                        <td className="text-text-muted text-[10px] max-w-[180px] truncate">
                          {alert.mitreTechniques?.join(", ") || "None"}
                        </td>
                        <td>
                          {alert.blockchainTxHash ? (
                            <span className="text-[10px] text-severity-normal flex items-center gap-1 font-bold">
                              <CheckCircle className="h-3 w-3" />
                              Recorded
                            </span>
                          ) : (
                            <span className="text-[10px] text-text-muted">Pending</span>
                          )}
                        </td>
                        <td className="text-right py-2.5">
                          <button
                            onClick={(e) => { e.stopPropagation(); verifyBlockchain(alert); }}
                            className="px-2.5 py-1 rounded bg-base-bg border border-base-border hover:border-accent/40 text-text-secondary hover:text-text-primary text-[10px] transition-colors"
                          >
                            Verify Audit
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ----------------- 6. AUDIT LOG TAB ----------------- */}
        {activeTab === "audit" && (
          <div className="bg-base-surface border border-base-border rounded-lg p-5">
            <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-accent" />
                <div>
                  <h3 className="font-bold uppercase text-xs tracking-wider font-mono-tech text-text-primary">
                    Tamper-Proof Blockchain Audit Ledger
                  </h3>
                  <p className="text-[10px] text-text-muted font-mono-tech">
                    Cryptographic Threat Forecast Verification (Smart Contract: ForecastRegistry)
                  </p>
                </div>
              </div>
              <span className="text-xs font-mono-tech text-severity-normal">
                {alerts.filter(a => a.blockchainTxHash).length} Verified Records
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono-tech text-xs">
                <thead>
                  <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px]">
                    <th className="py-2.5 font-semibold">Prediction ID</th>
                    <th className="font-semibold">Timestamp</th>
                    <th className="font-semibold">Target Asset</th>
                    <th className="font-semibold">Forecasted Threat</th>
                    <th className="font-semibold">Cryptographic Data Hash (SHA-256)</th>
                    <th className="font-semibold">Transaction Hash</th>
                    <th className="text-right font-semibold">Verification</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-base-border/50">
                  {alerts.filter(a => a.blockchainTxHash).length === 0 ? (
                    <tr>
                      <td colSpan="7" className="py-10 text-center text-text-muted">
                        No qualifying threat forecast has been generated. The cryptographic ledger records only non-benign threat events.
                      </td>
                    </tr>
                  ) : (
                    alerts.filter(a => a.blockchainTxHash).map(alert => (
                      <tr key={alert._id} className="hover:bg-base-surfaceHover/50 transition-colors">
                        <td className="py-2.5 font-mono-tech text-[10px] text-text-muted">{alert._id?.slice(0, 8)}...</td>
                        <td className="text-text-muted text-[10px]">{new Date(alert.timestamp).toLocaleString()}</td>
                        <td className="font-semibold text-text-primary">{alert.hostIp}</td>
                        <td>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getStageBadgeStyle(alert.predictedStage)}`}>
                            {alert.predictedStage}
                          </span>
                        </td>
                        <td className="text-[10px] text-text-muted font-mono-tech max-w-[160px] truncate">
                          {alert.dataHash || "d42f79f5fdbbd95f..."}
                        </td>
                        <td className="text-[10px] text-severity-normal font-mono-tech max-w-[160px] truncate">
                          {alert.blockchainTxHash}
                        </td>
                        <td className="text-right py-2.5">
                          <button
                            onClick={() => verifyBlockchain(alert)}
                            className="px-2.5 py-1 rounded bg-base-bg border border-base-border hover:border-accent text-accent text-[10px] transition-colors"
                          >
                            Verify Authenticity
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>

      {/* Alert Detail Slide-Out Drawer */}
      {selectedAlertForDrawer && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-xs">
          <div className="w-full max-w-md bg-base-surface border-l border-base-border h-full p-6 flex flex-col justify-between font-mono-tech shadow-2xl overflow-y-auto">
            <div>
              <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <ShieldAlert className="h-5 w-5 text-severity-critical" />
                  <div>
                    <h3 className="font-bold text-xs uppercase tracking-wider text-text-primary">
                      Threat Alert Forensics
                    </h3>
                    <p className="text-[10px] text-text-muted">ID: {selectedAlertForDrawer._id?.slice(0, 12)}...</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedAlertForDrawer(null)}
                  className="p-1 rounded text-text-muted hover:text-text-primary transition-colors"
                >
                  <XCircle className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div className="bg-base-bg p-3 rounded-lg border border-base-border space-y-2">
                  <div className="flex justify-between">
                    <span className="text-text-muted">Target Host:</span>
                    <span className="font-bold text-text-primary">{selectedAlertForDrawer.hostIp}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Severity:</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${selectedAlertForDrawer.severity === "CRITICAL" ? "bg-severity-critical/10 text-severity-critical" : "bg-severity-high/10 text-severity-high"}`}>
                      {selectedAlertForDrawer.severity}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Forecasted Stage:</span>
                    <span className="text-accent font-semibold">{selectedAlertForDrawer.predictedStage}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Threat Probability:</span>
                    <span className="text-text-primary font-bold">{(((selectedAlertForDrawer.threat_probability !== undefined ? selectedAlertForDrawer.threat_probability : (selectedAlertForDrawer.attack_probability !== undefined ? selectedAlertForDrawer.attack_probability : selectedAlertForDrawer.confidence)) || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Timestamp:</span>
                    <span className="text-text-secondary text-[10px]">{new Date(selectedAlertForDrawer.timestamp).toLocaleString()}</span>
                  </div>
                </div>

                {/* MITRE Techniques */}
                <div className="bg-base-bg p-3 rounded-lg border border-base-border">
                  <span className="text-[10px] text-text-muted block mb-1.5 uppercase font-bold">MITRE ATT&CK Techniques</span>
                  <div className="space-y-1">
                    {selectedAlertForDrawer.mitreTechniques && selectedAlertForDrawer.mitreTechniques.length > 0 ? (
                      selectedAlertForDrawer.mitreTechniques.map((t, idx) => (
                        <span key={idx} className="inline-block px-2 py-1 bg-accent/10 border border-accent/20 rounded text-[10px] text-accent font-medium mr-1.5 mb-1">
                          {t}
                        </span>
                      ))
                    ) : (
                      <span className="text-text-muted text-[10px]">No specific MITRE ATT&CK technique mapped.</span>
                    )}
                  </div>
                </div>

                {/* Snort Evidence */}
                <div className="bg-base-bg p-3 rounded-lg border border-base-border">
                  <span className="text-[10px] text-text-muted block mb-1.5 uppercase font-bold">Correlated Snort Signature Evidence</span>
                  {selectedAlertForDrawer.snortEvidence ? (
                    <div className="text-[10px] text-text-secondary space-y-1">
                      <div><strong>Signature:</strong> {selectedAlertForDrawer.snortEvidence.message}</div>
                      <div><strong>Priority:</strong> {selectedAlertForDrawer.snortEvidence.priority}</div>
                      <div><strong>Protocol:</strong> {selectedAlertForDrawer.snortEvidence.protocol}</div>
                    </div>
                  ) : (
                    <span className="text-text-muted text-[10px]">No local Snort signature alert correlated for this event.</span>
                  )}
                </div>

                {/* Blockchain Audit Status */}
                <div className="bg-base-bg p-3 rounded-lg border border-base-border">
                  <span className="text-[10px] text-text-muted block mb-1.5 uppercase font-bold">Cryptographic Ledger Provenance</span>
                  <div className="text-[10px] space-y-1">
                    <div className="truncate"><strong>Tx Hash:</strong> <span className="text-severity-normal">{selectedAlertForDrawer.blockchainTxHash || "Pending / Offline Node"}</span></div>
                    <div className="truncate"><strong>SHA-256 Hash:</strong> <span className="text-text-muted">{selectedAlertForDrawer.dataHash || "Computed"}</span></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-base-border space-y-2">
              <span className="text-[10px] text-text-muted block">TRIGGER MITIGATION ON HOST:</span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => { handleDefensiveAction(selectedAlertForDrawer.hostIp, "RATE_LIMIT"); setSelectedAlertForDrawer(null); }}
                  className="p-2 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-secondary text-[10px]"
                >
                  Rate Limit
                </button>
                <button
                  onClick={() => { handleDefensiveAction(selectedAlertForDrawer.hostIp, "ISOLATE"); setSelectedAlertForDrawer(null); }}
                  className="p-2 rounded bg-severity-critical/10 border border-severity-critical/30 hover:bg-severity-critical/20 text-severity-critical text-[10px] font-bold"
                >
                  Isolate Host
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Blockchain Verification Modal */}
      {verifyingAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm px-4">
          <div className="w-full max-w-lg bg-base-surface border border-base-border rounded-xl p-6 shadow-2xl relative font-mono-tech">
            <button
              onClick={() => { setVerifyingAlert(null); setVerificationResult(null); }}
              className="absolute top-4 right-4 text-text-muted hover:text-text-primary transition-colors"
            >
              <XCircle className="h-5 w-5" />
            </button>

            <div className="flex items-center gap-2 border-b border-base-border pb-3 mb-4">
              <Database className="h-5 w-5 text-accent" />
              <h3 className="font-bold uppercase text-xs tracking-wider text-text-primary">
                Cryptographic Audit Trail Verification
              </h3>
            </div>

            {loadingVerification ? (
              <div className="py-10 flex flex-col items-center justify-center gap-2 text-text-muted">
                <RefreshCw className="h-6 w-6 animate-spin text-accent" />
                <span className="text-xs">Reading record from decentralized node...</span>
              </div>
            ) : verificationResult ? (
              verificationResult.error ? (
                <div className="py-6 flex flex-col items-center gap-2 text-severity-critical">
                  <AlertTriangle className="h-8 w-8" />
                  <p className="text-xs text-center">{verificationResult.error}</p>
                </div>
              ) : (
                <div className="text-xs space-y-3.5">
                  {verificationResult.isAuthentic ? (
                    <div className="flex items-center gap-3 bg-severity-normal/10 border border-severity-normal/20 p-3.5 rounded-lg text-severity-normal">
                      <CheckCircle2 className="h-5 w-5 shrink-0" />
                      <div>
                        <h4 className="font-bold text-xs uppercase">Provenance Verification Complete</h4>
                        <p className="text-[10px] text-severity-normal/80 mt-0.5">
                          Cryptographic audit state matches local records. Forecast registration is authentic and tamper-evident.
                        </p>
                        <p className="text-[9px] text-text-muted mt-1 italic">
                          Proves record registration provenance in audit ledger; does not assert forecast ground-truth correctness.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3 bg-severity-high/10 border border-severity-high/20 p-3.5 rounded-lg text-severity-high">
                      <AlertTriangle className="h-5 w-5 shrink-0" />
                      <div>
                        <h4 className="font-bold text-xs uppercase">{verificationResult.status || "UNVERIFIED"}</h4>
                        <p className="text-[10px] text-severity-high/90 mt-0.5">
                          {verificationResult.message || "Forecast is not registered on-chain or local Ethereum node is offline."}
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="space-y-2 bg-base-bg/60 p-3.5 rounded-lg border border-base-border text-[11px]">
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Forecast ID</span>
                      <span className="text-text-primary font-medium">{verificationResult.alertId}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Host IP Address</span>
                      <span className="text-text-secondary">{verificationResult.blockchain?.hostIp || verificationResult.local?.hostIp || verifyingAlert?.hostIp}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Forecasted Threat Stage</span>
                      <span className="text-accent font-semibold">{verificationResult.blockchain?.predictedStage || verificationResult.local?.predictedStage || verifyingAlert?.predictedStage}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">On-Chain Block Number</span>
                      <span className="text-text-primary font-bold">{verificationResult.blockchain?.blockNumber ?? "N/A (Offline / Unregistered)"}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-text-muted">Cryptographic Data Hash</span>
                      <span className="text-severity-normal mt-1 select-all break-all bg-base-surface p-1.5 rounded border border-base-border text-[10px]">
                        {verificationResult.blockchain?.dataHash || verificationResult.local?.dataHash || verifyingAlert?.dataHash}
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-end gap-2 mt-4 pt-3 border-t border-base-border">
                    <button
                      onClick={() => { setVerifyingAlert(null); setVerificationResult(null); }}
                      className="px-4 py-1.5 bg-accent hover:bg-accent-hover text-white font-semibold text-[11px] rounded transition-colors"
                    >
                      CLOSE VIEW
                    </button>
                  </div>
                </div>
              )
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
