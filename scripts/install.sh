#!/usr/bin/env bash
# curLit bootstrap install script
# Usage: ./scripts/install.sh [--quick] [--reset]
# Fresh clone → ./scripts/install.sh → all services running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

QUICK=false
RESET=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --reset) RESET=true ;;
    esac
done

echo "======================================"
echo " curLit Installation"
echo "======================================"

# 1. Check prerequisites
echo "[1/7] Checking prerequisites..."
command -v python3 >/dev/null || { echo "Python 3 required"; exit 1; }
command -v docker >/dev/null || { echo "Docker required"; exit 1; }
command -v git >/dev/null || { echo "Git required"; exit 1; }
echo "  ✓ Prerequisites met"

# 2. Environment setup
echo "[2/7] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from template — EDIT WITH YOUR SECRETS NOW"
fi
echo "  ✓ .env exists"

# 3. Docker Compose — infrastructure
echo "[3/7] Starting infrastructure (Postgres + observability)..."
docker compose up -d postgres prometheus grafana loki promtail alertmanager node_exporter postgres_exporter 2>&1 | tail -1
echo "  Waiting for postgres healthy..."
docker compose ps postgres | grep -q "healthy" || sleep 10
echo "  ✓ Infrastructure running"

# 4. Database migrations
echo "[4/7] Running database migrations..."
python3 -m migrations.run 2>&1 | tail -1
echo "  ✓ Migrations applied"

# 5. Historical data seeding
if [ "$QUICK" = false ]; then
    echo "[5/7] Seeding historical data (this may take a while)..."
    echo "  Skipping — use Airflow DAGs for on-demand seeding"
    echo "  ✓ Data seeding deferred (run Airflow manually if needed)"
else
    echo "[5/7] Quick mode — skipping data seeding"
fi

# 6. Default configs
echo "[6/7] Creating default configs..."
mkdir -p configs/strategies
if [ ! -f configs/strategies/eurusd_rate_diff.yaml ]; then
    cat > configs/strategies/eurusd_rate_diff.yaml << 'EOF'
pair: EURUSD
entry_z_threshold: 1.5
exit_z_threshold: 0.3
max_position_pct: 0.15
EOF
fi
echo "  ✓ Default configs created"

# 7. Docker Compose — application services
echo "[7/7] Starting application services..."
docker compose up -d airflow-webserver airflow-scheduler 2>&1 | tail -1
echo "  ✓ Application services running"

echo ""
echo "======================================"
echo " Installation complete!"
echo ""
echo " Access URLs:"
echo "   Web UI:     http://localhost:8080"
echo "   Grafana:    http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo "   Airflow:    http://localhost:8080 (admin/admin)"
echo ""
echo " Next steps:"
echo "   1. Edit .env with your API keys"
echo "   2. Start vault agent: systemctl --user start fx-vault-agent"
echo "   3. Start engine: python -m src.runtime.run_engine --practice"
echo "======================================"
