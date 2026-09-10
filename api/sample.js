const https = require('https');
const http = require('http');

function fetchWithRedirect(url, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) {
      return reject(new Error('Too many redirects'));
    }

    const client = url.startsWith('https:') ? https : http;
    const req = client.get(url, { headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        let redirectUrl = res.headers.location;
        if (redirectUrl.startsWith('/')) {
          const parsed = new URL(url);
          redirectUrl = `${parsed.protocol}//${parsed.host}${redirectUrl}`;
        }
        return resolve(fetchWithRedirect(redirectUrl, maxRedirects - 1));
      }

      if (res.statusCode !== 200) {
        return reject(new Error(`Drive request failed with status ${res.statusCode}`));
      }

      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => {
        resolve({
          buffer: Buffer.concat(chunks),
          contentType: res.headers['content-type'] || 'audio/wav'
        });
      });
      res.on('error', reject);
    });

    req.on('error', reject);
  });
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const fileId = req.query.id || req.query.fileId;
  if (!fileId) {
    return res.status(400).json({ error: 'Missing required query parameter: id' });
  }

  // Sanitize fileId to prevent SSRF
  if (!/^[a-zA-Z0-9_-]{10,60}$/.test(fileId)) {
    return res.status(400).json({ error: 'Invalid Google Drive file ID format' });
  }

  const driveUrl = `https://drive.google.com/uc?export=download&id=${fileId}`;

  try {
    const data = await fetchWithRedirect(driveUrl);
    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Cache-Control', 'public, max-age=86400, s-maxage=86400');
    res.setHeader('Content-Length', data.buffer.length);
    return res.status(200).send(data.buffer);
  } catch (err) {
    console.error('Failed to proxy Drive file:', err.message);
    return res.status(502).json({
      error: 'Failed to fetch sample from Google Drive',
      details: err.message,
      fallback_url: driveUrl
    });
  }
};
