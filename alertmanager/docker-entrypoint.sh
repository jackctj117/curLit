#!/bin/sh
# Substitute env vars into alertmanager config template, then start Alertmanager.
set -e

sed \
  -e "s|__PUSHOVER_USER_KEY__|${PUSHOVER_USER_KEY}|g" \
  -e "s|__PUSHOVER_API_TOKEN__|${PUSHOVER_API_TOKEN}|g" \
  -e "s|__TELEGRAM_BOT_TOKEN__|${TELEGRAM_BOT_TOKEN}|g" \
  -e "s|__TELEGRAM_CHAT_ID__|${TELEGRAM_CHAT_ID}|g" \
  /etc/alertmanager/alertmanager.tmpl > /etc/alertmanager/alertmanager.yml

exec /bin/alertmanager \
  --config.file=/etc/alertmanager/alertmanager.yml \
  --storage.path=/alertmanager
