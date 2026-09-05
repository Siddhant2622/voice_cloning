module.exports = (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: "ok",
    deployment: "vercel-serverless",
    service: "VoiceGuard Detection System",
    engine: "Hybrid (Edge Web Audio + Cloud API)",
    models_loaded: true,
    cm_ready: true,
    timestamp: new Date().toISOString()
  });
};
