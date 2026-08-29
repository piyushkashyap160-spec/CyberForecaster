import React, { useState, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import { 
  ShieldAlert, Shield, Server, Cpu, Database, 
  Activity, CheckCircle, RefreshCw, AlertTriangle, Layers, Clock, 
  ExternalLink, CheckCircle2, XCircle, HardDrive
} from "lucide-react";
import { 
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from "recharts";

const API_BASE = "http://127.0.0.1:8000/api";
const SOCKET_URL = "http://127.0.0.1:8000";

export default function App() {
  const [hosts, setHosts] = useState([]);
  const [selectedHostIp, setSelectedHostIp] = useState("");
  const [alerts, setAlerts] = useState([]);
  const [trafficEvents, setTrafficEvents] = useState([]);
  const [rolloutData, setRolloutData] = useState(null);
  const [loadingRollout, setLoadingRollout] = useState(false);
  const [verifyingAlert, setVerifyingAlert] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [loadingVerification, setLoadingVerification] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [latestForecast, setLatestForecast] = useState({});

  // Collector UI State
  const [collectorInterfaces, setCollectorInterfaces] = useState([]);
  const [selectedInterface, setSelectedInterface] = useState("");
  const [collectorStatus, setCollectorStatus] = useState({ running: false, packets_captured: 0, flows_generated: 0, bytes_captured: 0 });
  const [loadingCollector, setLoadingCollector] = useState(false);

  const socketRef = useRef(null);

  // Load initial data
  useEffect(() => {
    fetchHosts();
    fetchAlerts();
    fetchCollectorInterfaces();
    fetchCollectorStatus();

    // Setup Socket.io Connection
    socketRef.current = io(SOCKET_URL);

    socketRef.current.on("connect", () => {
      setIsConnected(true);
      console.log("Connected to backend Socket.io server");
    });

    socketRef.current.on("disconnect", () => {
      setIsConnected(false);
    });

    socketRef.current.on("traffic_update", (event) => {
      setTrafficEvents(prev => [event, ...prev].slice(0, 30));
    });

    socketRef.current.on("forecast_update", (forecast) => {
      setLatestForecast(prev => ({
        ...prev,
        [forecast.hostIp]: forecast
      }));
    });

    socketRef.current.on("forecast_alert", (alert) => {
      setAlerts(prev => [alert, ...prev]);
      console.warn("Forecast Alert Triggered:", alert);
    });

    socketRef.current.on("host_status_change", ({ ip, status }) => {
      setHosts(prev => prev.map(h => h.ip === ip ? { ...h, status } : h));
    });

    return () => {
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  // Fetch hosts list
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

  // Fetch alerts
  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/alerts`);
      const data = await res.json();
      setAlerts(data);
    } catch (err) {
      console.error("Error fetching alerts:", err);
    }
  };

  // Collector API functions
  const fetchCollectorInterfaces = async () => {
    try {
      const res = await fetch(`${API_BASE}/collector/interfaces`);
      const data = await res.json();
      if (data.interfaces && data.interfaces.length > 0) {
        setCollectorInterfaces(data.interfaces);
        if (!selectedInterface) {
          const defaultIf = typeof data.interfaces[0] === 'string' ? data.interfaces[0] : (data.interfaces[0].name || "eth0");
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

  // Take defensive actions
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

  // Verify forecast on blockchain
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

  // Helper to format chart data
  const getChartData = () => {
    if (!rolloutData || !rolloutData.scenarios) return [];
    
    const stepsCount = rolloutData.rollout_steps || 6;
    const chartData = [];

    for (let t = 0; t < stepsCount; t++) {
      chartData.push({
        step: `T+${(t + 1) * 30}m`,
        "Do Nothing (Risk)": Math.round((rolloutData.scenarios.do_nothing?.[t]?.threat_level || 0) * 100),
        "Rate Limit": Math.round((rolloutData.scenarios.rate_limit?.[t]?.threat_level || 0) * 100),
        "Block Ports": Math.round((rolloutData.scenarios.block_port?.[t]?.threat_level || 0) * 100),
        "Isolate Host": Math.round((rolloutData.scenarios.isolate_host?.[t]?.threat_level || 0) * 100)
      });
    }
    return chartData;
  };

  const selectedHost = hosts.find(h => h.ip === selectedHostIp);
  const selectedHostForecast = latestForecast[selectedHostIp] || (alerts.find(a => a.hostIp === selectedHostIp));

  // Determine stage percentage
  const getStagePercentage = (stage) => {
    switch (stage) {
      case "Normal": return 10;
      case "Reconnaissance": return 30;
      case "Initial Access": return 55;
      case "Lateral Movement": return 75;
      case "Data Exfiltration": return 95;
      default: return 0;
    }
  };

  // Helper to get stage color
  const getStageTextColor = (stage) => {
    switch (stage) {
      case "Data Exfiltration": return "text-severity-critical";
      case "Lateral Movement": return "text-severity-high";
      case "Initial Access": return "text-severity-medium";
      case "Reconnaissance": return "text-accent";
      default: return "text-severity-normal";
    }
  };

  const getStageBadgeStyle = (stage) => {
    switch (stage) {
      case "Data Exfiltration": return "bg-severity-critical/10 border-severity-critical/20 text-severity-critical";
      case "Lateral Movement": return "bg-severity-high/10 border-severity-high/20 text-severity-high";
      case "Initial Access": return "bg-severity-medium/10 border-severity-medium/20 text-severity-medium";
      case "Reconnaissance": return "bg-accent/10 border-accent/20 text-accent";
      default: return "bg-severity-normal/10 border-severity-normal/20 text-severity-normal";
    }
  };

  return (
    <div className="min-h-screen dashboard-grid pb-12 font-sans select-none">
      {/* Header bar */}
      <header className="border-b border-base-border bg-base-surface/90 px-6 py-3.5 flex justify-between items-center sticky top-0 z-40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-md bg-accent/10 border border-accent/20 text-accent">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-wider text-text-primary font-mono-tech">
              CYBERFORECASTER <span className="text-text-muted font-normal">/ SOC WORLD MODEL</span>
            </h1>
            <p className="text-[11px] text-text-muted font-mono-tech">
              NTRO TACTICAL OPERATIONS · PS-26153
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono-tech">
          {/* Blockchain Audit Indicator */}
          <div className="flex items-center gap-2 bg-base-bg border border-severity-normal/20 px-3 py-1 rounded-full text-severity-normal">
            <div className="h-1.5 w-1.5 rounded-full bg-severity-normal pulse-indicator"></div>
            <span className="text-[11px]">ON-CHAIN AUDIT LOG ACTIVE</span>
          </div>

          {/* Connection status */}
          <div className={`flex items-center gap-2 bg-base-bg px-3 py-1 rounded-full border ${isConnected ? "border-base-border text-text-secondary" : "border-severity-critical/30 text-severity-critical"}`}>
            <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? "bg-severity-normal" : "bg-severity-critical"}`}></div>
            <span className="text-[11px]">{isConnected ? "ORCHESTRATOR ONLINE" : "DISCONNECTED"}</span>
          </div>
        </div>
      </header>

      {/* Top metrics ribbon */}
      <section className="grid grid-cols-4 gap-4 px-6 mt-5">
        {[
          { 
            title: "Total Monitored Hosts", 
            value: hosts.length, 
            icon: Server, 
            valueColor: "text-text-primary" 
          },
          { 
            title: "Active High-Threat Alerts", 
            value: alerts.filter(a => a.severity === "HIGH" || a.severity === "CRITICAL").length, 
            icon: ShieldAlert, 
            valueColor: alerts.some(a => a.severity === "HIGH" || a.severity === "CRITICAL") ? "text-severity-critical" : "text-text-primary" 
          },
          { 
            title: "Live Traffic Events", 
            value: trafficEvents.length, 
            icon: Activity, 
            valueColor: "text-accent" 
          },
          { 
            title: "On-Chain Predictions Logged", 
            value: alerts.filter(a => a.blockchainTxHash).length, 
            icon: Database, 
            valueColor: "text-severity-normal" 
          }
        ].map((item, idx) => (
          <div key={idx} className="bg-base-surface border border-base-border rounded-lg p-4 flex items-center justify-between card-panel-hover">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">{item.title}</p>
              <h2 className={`text-2xl font-bold mt-0.5 ${item.valueColor} font-mono-tech`}>{item.value}</h2>
            </div>
            <item.icon className="h-6 w-6 text-text-muted" />
          </div>
        ))}
      </section>

      {/* Main dashboard content */}
      <main className="grid grid-cols-12 gap-5 px-6 mt-5">
        
        {/* Left column: Topology and Live logs */}
        <section className="col-span-7 flex flex-col gap-5">
          
          {/* Network Topology */}
          <div className="bg-base-surface border border-base-border rounded-lg p-5 flex flex-col min-h-[360px]">
            <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3.5">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-accent" />
                <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">Subnet Topology Map</h3>
              </div>
              <span className="text-[10px] text-text-muted font-mono-tech">Configured Demo Subnet (192.168.1.0/24)</span>
            </div>

            {/* Topology SVG Canvas */}
            <div className="flex-1 relative flex items-center justify-center bg-base-bg/60 rounded-md border border-base-border/60 overflow-hidden">
              <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
                {/* Lines from Central Node to other nodes */}
                {hosts.map((h, idx) => {
                  const angle = (idx * 2 * Math.PI) / hosts.length;
                  const x = 50 + 32 * Math.cos(angle);
                  const y = 50 + 32 * Math.sin(angle);
                  
                  const isHighThreat = latestForecast[h.ip]?.predictedStage === "Data Exfiltration" || 
                                       latestForecast[h.ip]?.predictedStage === "Lateral Movement";
                  const lineColor = isHighThreat ? "rgba(240, 80, 110, 0.4)" : "#232838";
                  const strokeWidth = isHighThreat ? "1.5" : "1";
                  
                  return (
                    <line 
                      key={idx} 
                      x1="50%" 
                      y1="50%" 
                      x2={`${x}%`} 
                      y2={`${y}%`} 
                      stroke={lineColor}
                      strokeWidth={strokeWidth}
                      strokeDasharray={isHighThreat ? "3 3" : "0"}
                    />
                  );
                })}
              </svg>

              {/* Central Switch/Router node */}
              <div className="absolute z-10 flex flex-col items-center justify-center bg-base-surface border border-base-borderActive rounded-md p-2.5 shadow-md">
                <Cpu className="h-5 w-5 text-accent" />
                <span className="text-[9px] font-mono-tech mt-0.5 text-text-muted">GATEWAY</span>
              </div>

              {/* Dynamic Host Nodes placed in circular coordinates */}
              {hosts.map((h, idx) => {
                const angle = (idx * 2 * Math.PI) / hosts.length;
                const x = 50 + 32 * Math.cos(angle);
                const y = 50 + 32 * Math.sin(angle);

                const hostForecast = latestForecast[h.ip];
                const stage = hostForecast ? hostForecast.predictedStage : "Normal";
                const isSelected = h.ip === selectedHostIp;

                // Color code node status
                let borderClass = "border-base-border bg-base-surface text-text-primary";
                let iconColor = "text-text-muted";
                
                if (h.status === "ISOLATED") {
                  borderClass = "border-severity-critical/30 bg-severity-critical/10 text-severity-critical opacity-70";
                  iconColor = "text-severity-critical";
                } else if (stage === "Data Exfiltration") {
                  borderClass = "border-severity-critical/50 bg-severity-critical/10 text-severity-critical";
                  iconColor = "text-severity-critical";
                } else if (stage === "Lateral Movement" || stage === "Initial Access") {
                  borderClass = "border-severity-high/50 bg-severity-high/10 text-severity-high";
                  iconColor = "text-severity-high";
                } else if (stage === "Reconnaissance") {
                  borderClass = "border-accent/40 bg-accent/5 text-accent";
                  iconColor = "text-accent";
                } else {
                  if (isSelected) {
                    borderClass = "border-accent bg-accent/10 ring-1 ring-accent text-text-primary";
                    iconColor = "text-accent";
                  } else {
                    iconColor = "text-severity-normal";
                  }
                }

                return (
                  <button
                    key={h.ip}
                    onClick={() => setSelectedHostIp(h.ip)}
                    style={{ left: `${x}%`, top: `${y}%` }}
                    className={`absolute transform -translate-x-1/2 -translate-y-1/2 z-20 flex flex-col items-center p-2 rounded-md border transition-all duration-150 ${borderClass} ${isSelected ? "ring-1 ring-accent scale-105" : "hover:border-base-borderActive"}`}
                  >
                    <Server className={`h-4 w-4 ${iconColor}`} />
                    <span className="text-[10px] font-mono-tech mt-1 font-medium truncate max-w-[80px]">{h.name}</span>
                    <span className="text-[8px] text-text-muted font-mono-tech">{h.ip}</span>
                    
                    {/* Mitigation badge */}
                    {h.status !== "ONLINE" && (
                      <span className="absolute -top-1.5 -right-1.5 text-[7px] bg-severity-critical text-white font-bold px-1 rounded uppercase">
                        {h.status.replace("_", " ")}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Live Network Capture Control Panel */}
          <div className="bg-base-surface border border-base-border rounded-lg p-4">
            <div className="flex justify-between items-center border-b border-base-border pb-2.5 mb-3">
              <div className="flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-accent" />
                <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">Live Network Packet Collector</h3>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-mono-tech px-2 py-0.5 rounded border ${collectorStatus.running ? "bg-severity-normal/10 border-severity-normal/20 text-severity-normal" : "bg-base-bg border-base-border text-text-muted"}`}>
                  {collectorStatus.running ? "CAPTURING LIVE" : "COLLECTOR IDLE"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-3 items-center text-xs font-mono-tech">
              <div className="col-span-5 flex items-center gap-1.5">
                <select
                  value={selectedInterface}
                  onChange={(e) => setSelectedInterface(e.target.value)}
                  className="w-full bg-base-bg border border-base-border rounded px-2.5 py-1.5 text-text-primary focus:outline-none focus:border-accent text-xs font-mono-tech"
                >
                  {collectorInterfaces.length === 0 ? (
                    <option value="eth0">eth0 (Default)</option>
                  ) : (
                    collectorInterfaces.map((iface, idx) => {
                      const val = typeof iface === 'string' ? iface : (iface.name || `iface-${idx}`);
                      const desc = typeof iface === 'string' ? iface : (iface.description || iface.name);
                      return <option key={idx} value={val}>{desc}</option>;
                    })
                  )}
                </select>
                <button
                  onClick={fetchCollectorInterfaces}
                  title="Refresh Interfaces"
                  className="p-1.5 rounded bg-base-bg border border-base-border hover:bg-base-surfaceHover text-text-secondary transition-colors"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="col-span-4 flex gap-2">
                {!collectorStatus.running ? (
                  <button
                    onClick={startCollector}
                    disabled={loadingCollector}
                    className="w-full py-1.5 px-3 rounded bg-severity-normal/15 border border-severity-normal/30 hover:bg-severity-normal/25 text-severity-normal font-semibold text-xs transition-colors disabled:opacity-40"
                  >
                    {loadingCollector ? "STARTING..." : "START CAPTURE"}
                  </button>
                ) : (
                  <button
                    onClick={stopCollector}
                    disabled={loadingCollector}
                    className="w-full py-1.5 px-3 rounded bg-severity-critical/15 border border-severity-critical/30 hover:bg-severity-critical/25 text-severity-critical font-semibold text-xs transition-colors disabled:opacity-40"
                  >
                    {loadingCollector ? "STOPPING..." : "STOP CAPTURE"}
                  </button>
                )}
              </div>

              <div className="col-span-3 flex justify-end gap-3 text-[11px] text-text-muted">
                <div>Pkts: <span className="text-text-primary font-bold">{collectorStatus.packets_captured || 0}</span></div>
                <div>Flows: <span className="text-accent font-bold">{collectorStatus.flows_generated || 0}</span></div>
              </div>
            </div>
          </div>

          {/* Live Packet Streaming */}
          <div className="bg-base-surface border border-base-border rounded-lg p-4 flex-1 flex flex-col max-h-[300px]">
            <div className="flex justify-between items-center border-b border-base-border pb-2.5 mb-3">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-accent" />
                <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">Live Network Traffic Flows</h3>
              </div>
              <span className="text-[10px] text-text-muted font-mono-tech">Streaming Buffer</span>
            </div>

            <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-1.5 font-mono-tech text-[11px]">
              {trafficEvents.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-10 text-text-muted gap-2">
                  <Activity className="h-6 w-6 opacity-30 animate-spin" />
                  <span className="text-xs">Awaiting network traffic simulation...</span>
                </div>
              ) : (
                trafficEvents.map((evt, idx) => {
                  const dateStr = new Date(evt.timestamp).toLocaleTimeString();
                  
                  let actionLabel = "NONE";
                  let actionColor = "text-text-muted";
                  if (evt.action === 1) { actionLabel = "RATE LIMIT"; actionColor = "text-severity-high"; }
                  else if (evt.action === 2) { actionLabel = "PORTS BLOCKED"; actionColor = "text-severity-high"; }
                  else if (evt.action === 3) { actionLabel = "ISOLATED"; actionColor = "text-severity-critical font-semibold"; }

                  return (
                    <div key={idx} className="flex justify-between bg-base-bg/50 border border-base-border/50 rounded px-2.5 py-1.5 hover:bg-base-surfaceHover transition-colors">
                      <div className="flex gap-3">
                        <span className="text-text-muted">{dateStr}</span>
                        <span className="text-text-primary font-medium">{evt.hostIp}</span>
                        <span className="text-text-secondary">Dur: {evt.duration.toFixed(2)}s</span>
                        <span className="text-text-secondary">Bytes: {evt.total_bytes.toLocaleString()}</span>
                      </div>
                      <div className="flex gap-3">
                        <span className="text-text-muted font-medium">{evt.protocol === 1 ? "TCP" : evt.protocol === 0.5 ? "UDP" : "ICMP"}</span>
                        <span className={`${actionColor}`}>[ACT: {actionLabel}]</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </section>

        {/* Right column: World Model Forecasting Details */}
        <section className="col-span-5 flex flex-col gap-5">
          <div className="bg-base-surface border border-base-border rounded-lg p-5 flex-1 flex flex-col">
            <div className="flex justify-between items-center border-b border-base-border pb-3 mb-4">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-accent" />
                <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">Proactive World Model Forecaster</h3>
              </div>
              <span className="text-[10px] text-accent font-mono-tech bg-accent/10 px-2 py-0.5 rounded border border-accent/20">ACTIVE HOST</span>
            </div>

            {selectedHost ? (
              <div className="flex-1 flex flex-col gap-4">
                
                {/* Host metadata summary */}
                <div className="flex justify-between items-start bg-base-bg/60 border border-base-border rounded-lg p-3.5">
                  <div>
                    <h4 className="font-bold text-text-primary text-sm">{selectedHost.name}</h4>
                    <p className="text-xs text-text-secondary mt-0.5 font-mono-tech">IP: {selectedHost.ip}</p>
                    <p className="text-[10px] text-text-muted font-mono-tech uppercase mt-0.5">Dept: {selectedHost.department}</p>
                  </div>
                  <div className="text-right">
                    <span className={`text-[10px] font-bold font-mono-tech px-2 py-0.5 rounded border ${
                      selectedHost.criticality === "CRITICAL" ? "bg-severity-critical/10 border-severity-critical/20 text-severity-critical" :
                      selectedHost.criticality === "HIGH" ? "bg-severity-high/10 border-severity-high/20 text-severity-high" :
                      "bg-base-surface border-base-border text-text-muted"
                    }`}>
                      {selectedHost.criticality} CRITICALITY
                    </span>
                    <p className="text-[10px] mt-1.5 text-text-muted font-mono-tech uppercase">Status: 
                      <span className={`font-semibold ml-1.5 ${selectedHost.status === "ONLINE" ? "text-severity-normal" : "text-severity-critical"}`}>
                        {selectedHost.status}
                      </span>
                    </p>
                  </div>
                </div>

                {/* Predictor Panel */}
                <div className="border border-base-border p-3.5 rounded-lg bg-base-bg/40">
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech mb-2">Forecasted Attacker Stage</h4>
                  
                  {selectedHostForecast ? (
                    <div>
                      <div className="flex justify-between items-baseline">
                        <span className={`text-lg font-bold font-mono-tech uppercase ${getStageTextColor(selectedHostForecast.predictedStage)}`}>
                          {selectedHostForecast.predictedStage}
                        </span>
                        <span className="text-xs text-text-secondary font-mono-tech">
                          Confidence: <span className="font-semibold text-text-primary">{(selectedHostForecast.confidence * 100).toFixed(1)}%</span>
                        </span>
                      </div>

                      {/* Timeline progression bar */}
                      <div className="mt-2.5">
                        <div className="relative w-full h-1.5 bg-base-surface rounded-full overflow-hidden border border-base-border/40">
                          <div 
                            className={`h-full rounded-full transition-all duration-700 ${
                              selectedHostForecast.predictedStage === "Data Exfiltration" ? "bg-severity-critical" :
                              selectedHostForecast.predictedStage === "Lateral Movement" ? "bg-severity-high" :
                              selectedHostForecast.predictedStage === "Initial Access" ? "bg-severity-medium" :
                              selectedHostForecast.predictedStage === "Normal" ? "bg-severity-normal" : "bg-accent"
                            }`}
                            style={{ width: `${getStagePercentage(selectedHostForecast.predictedStage)}%` }}
                          ></div>
                        </div>
                        <div className="flex justify-between text-[8px] font-mono-tech mt-1 text-text-muted uppercase">
                          <span>Normal</span>
                          <span>Recon</span>
                          <span>Initial Access</span>
                          <span>Lateral Move</span>
                          <span>Exfil</span>
                        </div>
                      </div>

                      {/* MITRE Mapping */}
                      {selectedHostForecast.mitreTechniques && selectedHostForecast.mitreTechniques.length > 0 && (
                        <div className="mt-3 border-t border-base-border/60 pt-2.5">
                          <p className="text-[10px] text-text-muted uppercase font-mono-tech mb-1">MITRE ATT&CK Techniques Implicated:</p>
                          <div className="flex flex-wrap gap-1">
                            {selectedHostForecast.mitreTechniques.map((tech, idx) => (
                              <span key={idx} className="text-[9px] font-mono-tech bg-base-surface text-text-secondary px-2 py-0.5 rounded border border-base-border">
                                {tech}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-text-muted font-mono-tech py-3 text-center">
                      No anomalies forecasted. Subnet behaves normally.
                    </div>
                  )}
                </div>

                {/* Defensive Action mitigation panel */}
                <div className="border border-base-border p-3.5 rounded-lg bg-base-bg/40">
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech mb-2.5">Defensive Interventions</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono-tech">
                    <button 
                      onClick={() => handleDefensiveAction(selectedHost.ip, "RATE_LIMIT")}
                      disabled={selectedHost.status === "RATE_LIMITED" || selectedHost.status === "ISOLATED"}
                      className="flex items-center justify-center gap-1.5 p-2 rounded bg-base-surface border border-base-border hover:border-base-borderActive hover:bg-base-surfaceHover text-text-primary transition-colors disabled:opacity-30"
                    >
                      <Clock className="h-3.5 w-3.5 text-severity-high" />
                      <span>RATE LIMIT FLOWS</span>
                    </button>
                    <button 
                      onClick={() => handleDefensiveAction(selectedHost.ip, "BLOCK_PORTS")}
                      disabled={selectedHost.status === "PORTS_BLOCKED" || selectedHost.status === "ISOLATED"}
                      className="flex items-center justify-center gap-1.5 p-2 rounded bg-base-surface border border-base-border hover:border-base-borderActive hover:bg-base-surfaceHover text-text-primary transition-colors disabled:opacity-30"
                    >
                      <Shield className="h-3.5 w-3.5 text-severity-high" />
                      <span>BLOCK PORT CHANNELS</span>
                    </button>
                    <button 
                      onClick={() => handleDefensiveAction(selectedHost.ip, "ISOLATE")}
                      disabled={selectedHost.status === "ISOLATED"}
                      className="col-span-2 flex items-center justify-center gap-1.5 p-2 rounded bg-severity-critical/10 border border-severity-critical/30 hover:bg-severity-critical/20 text-severity-critical font-semibold transition-colors disabled:opacity-30"
                    >
                      <ShieldAlert className="h-4 w-4 text-severity-critical" />
                      <span>ISOLATE HOST (QUARANTINE)</span>
                    </button>
                    {selectedHost.status !== "ONLINE" && (
                      <button 
                        onClick={() => handleDefensiveAction(selectedHost.ip, "RESET")}
                        className="col-span-2 flex items-center justify-center gap-1.5 p-2 rounded bg-base-surface border border-base-border hover:bg-base-surfaceHover text-text-secondary hover:text-text-primary mt-0.5 transition-colors"
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        <span>RECONNECT & RESET HOST STATE</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* What-If Forecasting Plot */}
                <div className="flex-1 border border-base-border p-3.5 rounded-lg bg-base-bg/40 flex flex-col min-h-[220px]">
                  <div className="flex justify-between items-baseline mb-2">
                    <h4 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary font-mono-tech">"What-If" Rollout Forecasting (6-Step Horizon)</h4>
                    <span className="text-[8px] text-text-muted font-mono-tech italic">illustrative mitigation impact (heuristic, not model-learned)</span>
                  </div>
                  
                  {loadingRollout ? (
                    <div className="flex-1 flex flex-col items-center justify-center py-6 text-text-muted gap-2">
                      <RefreshCw className="h-5 w-5 animate-spin text-accent" />
                      <span className="text-xs font-mono-tech">Simulating counterfactual scenarios...</span>
                    </div>
                  ) : rolloutData ? (
                    <div className="flex-1 min-h-[150px] text-[10px]">
                      <ResponsiveContainer width="100%" height={150}>
                        <AreaChart data={getChartData()} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
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
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-xs text-text-muted py-6 font-mono-tech text-center">
                      Awaiting network traffic simulation history to project future rollouts.
                    </div>
                  )}
                </div>

              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-text-muted py-10">
                No host selected.
              </div>
            )}
          </div>
        </section>

      </main>

      {/* Bottom Section: Active Security Alerts & On-chain verify log */}
      <section className="px-6 mt-5">
        <div className="bg-base-surface border border-base-border rounded-lg p-5">
          <div className="flex justify-between items-center border-b border-base-border pb-3 mb-3.5">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-severity-critical" />
              <h3 className="font-semibold uppercase text-xs tracking-wider text-text-secondary font-mono-tech">Triggered Attack Forecast Logs</h3>
            </div>
            <span className="text-[10px] text-text-muted font-mono-tech uppercase">Tamper-Proof Blockchain Audit Logs Integrated</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono-tech text-xs">
              <thead>
                <tr className="border-b border-base-border text-text-secondary uppercase tracking-wider text-[10px] pb-2">
                  <th className="py-2.5 font-semibold">Alert Timestamp</th>
                  <th className="font-semibold">Target IP</th>
                  <th className="font-semibold">Forecasted Stage</th>
                  <th className="font-semibold">Confidence</th>
                  <th className="font-semibold">Identified MITRE Tactics</th>
                  <th className="font-semibold">Blockchain Status</th>
                  <th className="text-right font-semibold">Action Trail</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-base-border/50">
                {alerts.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="py-6 text-center text-text-muted">
                      No active threat forecasting alerts recorded. Network remains secure.
                    </td>
                  </tr>
                ) : (
                  alerts.map((alert) => (
                    <tr key={alert._id} className="hover:bg-base-surfaceHover/50 transition-colors">
                      <td className="py-2.5 text-text-muted">
                        {new Date(alert.timestamp).toLocaleString()}
                      </td>
                      <td className="font-semibold text-text-primary">{alert.hostIp}</td>
                      <td>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-medium border ${getStageBadgeStyle(alert.predictedStage)}`}>
                          {alert.predictedStage}
                        </span>
                      </td>
                      <td className="text-text-secondary">{(alert.confidence * 100).toFixed(1)}%</td>
                      <td className="text-text-muted text-[10px] max-w-[200px] truncate">
                        {alert.mitreTechniques?.join(", ") || "None"}
                      </td>
                      <td>
                        {alert.blockchainTxHash ? (
                          <div className="flex items-center gap-1 text-severity-normal">
                            <CheckCircle className="h-3.5 w-3.5" />
                            <span className="text-[10px] truncate max-w-[120px]">{alert.blockchainTxHash}</span>
                          </div>
                        ) : (
                          <span className="text-text-muted">Not Logged</span>
                        )}
                      </td>
                      <td className="text-right py-2.5">
                        <button 
                          onClick={() => verifyBlockchain(alert)}
                          className="px-2.5 py-1 rounded bg-base-bg border border-base-border hover:border-accent/40 text-text-secondary hover:text-text-primary text-[10px] transition-colors flex items-center gap-1.5 ml-auto"
                        >
                          <Database className="h-3 w-3 text-severity-normal" />
                          <span>Verify Audit Trail</span>
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Blockchain Verification Modal */}
      {verifyingAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm px-4">
          <div className="w-full max-w-lg bg-base-surface border border-base-border rounded-xl p-6 shadow-2xl relative">
            <button 
              onClick={() => { setVerifyingAlert(null); setVerificationResult(null); }}
              className="absolute top-4 right-4 text-text-muted hover:text-text-primary transition-colors"
            >
              <XCircle className="h-5 w-5" />
            </button>

            <div className="flex items-center gap-2 border-b border-base-border pb-3 mb-4">
              <Database className="h-5 w-5 text-accent" />
              <h3 className="font-bold uppercase text-xs tracking-wider font-mono-tech text-text-primary">
                Cryptographic Audit Trail Verification
              </h3>
            </div>

            {loadingVerification ? (
              <div className="py-10 flex flex-col items-center justify-center gap-2 text-text-muted">
                <RefreshCw className="h-6 w-6 animate-spin text-accent" />
                <span className="text-xs font-mono-tech">Reading record from decentralized node...</span>
              </div>
            ) : verificationResult ? (
              verificationResult.error ? (
                <div className="py-6 flex flex-col items-center gap-2 text-severity-critical">
                  <AlertTriangle className="h-8 w-8" />
                  <p className="text-xs text-center font-mono-tech">{verificationResult.error}</p>
                </div>
              ) : (
                <div className="font-mono-tech text-xs space-y-3.5">
                  <div className="flex items-center gap-3 bg-severity-normal/10 border border-severity-normal/20 p-3.5 rounded-lg text-severity-normal">
                    <CheckCircle2 className="h-5 w-5 shrink-0" />
                    <div>
                      <h4 className="font-bold text-xs uppercase">Verification Complete</h4>
                      <p className="text-[10px] text-severity-normal/80 mt-0.5">
                        Cryptographic state matches local records. Prediction is authentic and tamper-proof.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2 bg-base-bg/60 p-3.5 rounded-lg border border-base-border text-[11px]">
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Forecast ID</span>
                      <span className="text-text-primary font-medium">{verificationResult.alertId}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Host IP Address</span>
                      <span className="text-text-secondary">{verificationResult.blockchain.hostIp}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">Forecasted Threat Stage</span>
                      <span className="text-accent font-semibold">{verificationResult.blockchain.predictedStage}</span>
                    </div>
                    <div className="flex justify-between border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">On-Chain Block Number</span>
                      <span className="text-text-primary font-bold">{verificationResult.blockchain.blockNumber}</span>
                    </div>
                    <div className="flex flex-col border-b border-base-border/50 pb-1.5">
                      <span className="text-text-muted">On-Chain Block Timestamp</span>
                      <span className="text-text-secondary mt-0.5">
                        {new Date(verificationResult.blockchain.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-text-muted">Cryptographic Data Hash</span>
                      <span className="text-severity-normal mt-1 select-all break-all bg-base-surface p-1.5 rounded border border-base-border text-[10px]">
                        {verificationResult.blockchain.dataHash}
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-end gap-2 mt-4 pt-3 border-t border-base-border">
                    <a 
                      href={`http://127.0.0.1:8545`} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="px-3.5 py-1.5 bg-base-bg border border-base-border hover:bg-base-surfaceHover text-[11px] text-text-secondary hover:text-text-primary rounded transition-colors flex items-center gap-1.5"
                    >
                      <span>Transaction Ledger</span>
                      <ExternalLink className="h-3 w-3" />
                    </a>
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
