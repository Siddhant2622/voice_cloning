const crypto = require('crypto');

const DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];

module.exports = (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Generate 6 random digits
  const digits = Array.from({ length: 6 }, () => Math.floor(Math.random() * 10));
  const digitStr = digits.join(" ");
  const phrase = digits.map(d => DIGIT_WORDS[d]).join(" ");
  const challengeId = "ch_" + crypto.randomBytes(6).toString('hex');

  return res.status(200).json({
    challenge_id: challengeId,
    phrase: phrase,
    digits: digitStr,
    expires_in_s: 30,
    created_at: new Date().toISOString()
  });
};
