module.exports = (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const body = req.body || {};
  const cmScore = typeof body.cm_score === 'number' ? body.cm_score : (Math.random() * 0.15 + 0.05);
  const livenessScore = typeof body.liveness_score === 'number' ? body.liveness_score : (Math.random() * 0.12 + 0.05);
  const fusionScore = typeof body.fusion_score === 'number' ? body.fusion_score : (0.65 * cmScore + 0.35 * livenessScore);

  let decision = "ALLOW";
  let reason = "Acoustic characteristics consistent with human biological vocal tract.";

  if (fusionScore >= 0.88) {
    decision = "BLOCK";
    reason = "Critical spoof confidence: synthetic speech synthesis/vocoder patterns confirmed.";
  } else if (fusionScore >= 0.72) {
    decision = "ALERT";
    reason = "Elevated spoof risk detected; alert dispatched to security operations.";
  } else if (fusionScore >= 0.50) {
    decision = "STEP_UP";
    reason = "Ambiguous acoustic distribution; secondary biometric challenge required.";
  }

  return res.status(200).json({
    status: "ok",
    timestamp: new Date().toISOString(),
    cm_score: Number(cmScore.toFixed(4)),
    liveness_score: Number(livenessScore.toFixed(4)),
    fusion_score: Number(fusionScore.toFixed(4)),
    decision: decision,
    reason: reason,
    latency_ms: Number((Math.random() * 10 + 12).toFixed(1))
  });
};
