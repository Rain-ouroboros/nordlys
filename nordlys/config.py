SR = 44100
BLOCK_SEC = 8
BLOCK = SR * BLOCK_SEC
PORT = 8799
LEAD_SEC = 6.0
MP3_BITRATE = "128k"
SPOOL_DIR = "spool/voice"
TTS_CMD = "espeak-ng -v ru -s 140 -w {out} {text}"
