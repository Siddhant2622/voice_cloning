module.exports = (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const sampleLogs = [
    {
      timestamp: new Date(Date.now() - 360000).toISOString(),
      window_index: 12,
      cm_score: 0.124,
      liveness_score: 0.082,
      fusion_score: 0.108,
      decision: "ALLOW",
      reason: "Low risk speech detected; acoustic features consistent with genuine vocal tract.",
      latency_ms: 18.2
    },
    {
      timestamp: new Date(Date.now() - 180000).toISOString(),
      window_index: 24,
      cm_score: 0.941,
      liveness_score: 0.887,
      fusion_score: 0.923,
      decision: "BLOCK",
      reason: "High synthetic voice probability detected (neural vocoder artifacts identified).",
      latency_ms: 22.6
    },
    {
      timestamp: new Date(Date.now() - 60000).toISOString(),
      window_index: 38,
      cm_score: 0.612,
      liveness_score: 0.540,
      fusion_score: 0.589,
      decision: "STEP_UP",
      reason: "Unusual spectral flatness and pitch contour anomalies; step-up verification triggered.",
      latency_ms: 19.4
    }
  ];

  return res.status(200).json({ logs: sampleLogs });
};
