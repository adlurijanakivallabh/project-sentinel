// ============================================
// Project Sentinel — IPS Dashboard v2.0
// Real-time IPS monitoring with prevention
// ============================================

Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', sans-serif";

// 8-Class V2 color scheme
const CLASS_COLORS = {
    'Normal':          '#34d399',
    'DoS/DDoS':        '#f87171',
    'PortScan/Recon':  '#a78bfa',
    'Web/Injection':   '#f472b6',
    'Brute Force':     '#e879f9',
    'Botnet/C2':       '#38bdf8',
    'Malware/Exploit': '#22d3ee',
    'Infiltration':    '#facc15',
};

const CLASS_LABELS = ['Normal', 'DoS/DDoS', 'PortScan/Recon', 'Web/Injection', 'Brute Force', 'Botnet/C2', 'Malware/Exploit', 'Infiltration'];

// ─── Elements ───
const packetsEl     = document.getElementById('totalPackets');
const threatsEl     = document.getElementById('threatsDetected');
const blockedEl     = document.getElementById('ipsBlocked');
const latencyEl     = document.getElementById('latencyStat');
const confidenceEl  = document.getElementById('confidenceStat');
const feedBody      = document.getElementById('feedBody');
const statusMsg     = document.getElementById('systemStatusMsg');
const feedCount     = document.getElementById('feedCount');
const ipsActionsList = document.getElementById('ipsActionsList');
const blockedIpsList = document.getElementById('blockedIpsList');
const ipsToggle     = document.getElementById('ipsToggle');
const ipsHealthBadge = document.getElementById('ipsHealthBadge');

// ─── Line Chart ───
const ctxLine = document.getElementById('liveTrafficChart').getContext('2d');
const lineGradient = ctxLine.createLinearGradient(0, 0, 0, 300);
lineGradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
lineGradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

const liveTrafficChart = new Chart(ctxLine, {
    type: 'line',
    data: {
        labels: Array.from({length: 30}, (_, i) => ''),
        datasets: [{
            data: Array(30).fill(8),
            borderColor: '#818cf8',
            backgroundColor: lineGradient,
            borderWidth: 2, tension: 0.4, fill: true,
            pointRadius: 0, pointHoverRadius: 4,
            pointHoverBackgroundColor: '#818cf8',
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: {
                beginAtZero: true, max: 15,
                grid: { color: 'rgba(255,255,255,0.03)' },
                ticks: { font: { family: "'JetBrains Mono'", size: 10 } },
            },
            x: { grid: { display: false }, ticks: { display: false } },
        },
        animation: { duration: 0 },
        interaction: { intersect: false, mode: 'index' },
    }
});

// ─── Doughnut Chart ───
const ctxDoughnut = document.getElementById('threatPieChart').getContext('2d');
const threatPieChart = new Chart(ctxDoughnut, {
    type: 'doughnut',
    data: {
        labels: CLASS_LABELS,
        datasets: [{
            data: [1, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: CLASS_LABELS.map(l => CLASS_COLORS[l]),
            borderWidth: 0, hoverOffset: 8,
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        cutout: '72%',
        plugins: {
            legend: {
                position: 'right',
                labels: {
                    padding: 12, color: '#94a3b8',
                    font: { size: 11, family: "'Inter'" },
                    usePointStyle: true, pointStyleWidth: 8,
                }
            }
        },
    }
});

// ─── API Controls ───
async function sendCommand(action, type = null) {
    await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, type }),
    });
}

document.getElementById('btnStart').onclick    = () => sendCommand('start');
document.getElementById('btnStop').onclick     = () => sendCommand('stop');
document.getElementById('btnNormal').onclick   = () => sendCommand('inject', null);
document.getElementById('btnDoS').onclick      = () => sendCommand('inject', 'DoS-Hulk');
document.getElementById('btnScan').onclick     = () => sendCommand('inject', 'PortScan');
document.getElementById('btnSql').onclick      = () => sendCommand('inject', 'SQL-Injection');
document.getElementById('btnBrute').onclick    = () => sendCommand('inject', 'FTP-Brute');
document.getElementById('btnBotnet').onclick   = () => sendCommand('inject', 'Botnet');
document.getElementById('btnExploit').onclick  = () => sendCommand('inject', 'Exploit');
document.getElementById('btnInfil').onclick    = () => sendCommand('inject', 'Infiltration');
document.getElementById('btnUnblockAll').onclick = () => sendCommand('unblock_all');

ipsToggle.onclick = () => {
    sendCommand('toggle_ips');
    ipsToggle.classList.toggle('active');
    const isActive = ipsToggle.classList.contains('active');
    ipsToggle.textContent = isActive ? 'ON' : 'OFF';
    ipsHealthBadge.textContent = isActive ? 'ACTIVE' : 'DISABLED';
};

// ─── Model Selector ───
const modelSelector = document.getElementById('modelSelector');
const modelBadge = document.getElementById('modelBadge');

modelSelector.addEventListener('change', async () => {
    const variant = modelSelector.value;
    modelSelector.classList.add('loading');
    modelBadge.textContent = '...';
    modelBadge.style.color = '#fbbf24';
    try {
        const res = await fetch('/api/switch_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ variant }),
        });
        const data = await res.json();
        if (data.status === 'ok') {
            modelBadge.textContent = data.params;
            modelBadge.style.color = '#6ee7b7';
        } else {
            modelBadge.textContent = 'ERR';
            modelBadge.style.color = '#f87171';
            alert('Model switch failed: ' + data.message);
        }
    } catch (e) {
        modelBadge.textContent = 'ERR';
        modelBadge.style.color = '#f87171';
    }
    modelSelector.classList.remove('loading');
});

// Color inject buttons
document.querySelectorAll('.inject-btn').forEach(btn => {
    const color = btn.getAttribute('data-color');
    if (color) {
        btn.addEventListener('mouseenter', () => {
            btn.style.borderColor = color;
            btn.style.color = color;
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.borderColor = 'rgba(255,255,255,0.1)';
            btn.style.color = '#94a3b8';
        });
    }
});

// ─── Format numbers ───
function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}

// ─── Live Capture Controls ───
const liveCaptureToggle = document.getElementById('liveCaptureToggle');
const liveCaptureHealthBadge = document.getElementById('liveCaptureHealthBadge');
const liveCaptureStats = document.getElementById('liveCaptureStats');
let liveModeActive = false;

liveCaptureToggle.onclick = async () => {
    try {
        const action = liveModeActive ? 'stop' : 'start';
        const res = await fetch('/api/live_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action }),
        });
        const data = await res.json();
        if (data.status === 'ok') {
            liveModeActive = data.live_mode;
            if (liveModeActive) {
                liveCaptureToggle.textContent = 'ON';
                liveCaptureToggle.classList.add('active', 'live-active');
                liveCaptureHealthBadge.textContent = 'ACTIVE';
                liveCaptureHealthBadge.style.color = '#67e8f9';
                liveCaptureStats.style.display = 'block';
            } else {
                liveCaptureToggle.textContent = 'OFF';
                liveCaptureToggle.classList.remove('active', 'live-active');
                liveCaptureHealthBadge.textContent = 'OFF';
                liveCaptureHealthBadge.style.color = '#64748b';
                liveCaptureStats.style.display = 'none';
            }
        } else {
            alert('Live capture error: ' + (data.message || 'Unknown error'));
        }
    } catch (e) {
        console.error('Live capture toggle failed:', e);
    }
};

// Poll live capture stats every 2s when active
async function updateLiveStats() {
    if (!liveModeActive) return;
    try {
        const res = await fetch('/api/live_status');
        const data = await res.json();
        if (data.stats) {
            document.getElementById('lcPackets').textContent = formatNumber(data.stats.packets_captured || 0);
            document.getElementById('lcFlows').textContent = formatNumber(data.stats.flows_created || 0);
            document.getElementById('lcActive').textContent = data.active_flows || 0;
            document.getElementById('lcWindows').textContent = data.stats.windows_classified || 0;
            document.getElementById('lcAttacks').textContent = data.stats.attacks_detected || 0;
        }
    } catch (e) { /* ignore */ }
}
setInterval(updateLiveStats, 2000);

// ─── Dashboard Update Loop ───
let lastAlertCount = 0;

async function updateDashboard() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // KPI Cards
        packetsEl.textContent = formatNumber(data.stats.total);
        threatsEl.textContent = formatNumber(data.stats.attacks);
        blockedEl.textContent = data.ips.blocked_count;
        latencyEl.textContent = data.stats.latency.toFixed(2) + ' ms';

        // Confidence — show latest ATTACK confidence (skip Normal events)
        const latestAttack = data.alerts.find(a => a.meta_class !== 'Normal');
        if (latestAttack) {
            confidenceEl.textContent = latestAttack.confidence + '%';
            const conf = parseFloat(latestAttack.confidence);
            if (conf >= 90) confidenceEl.style.color = '#34d399';
            else if (conf >= 70) confidenceEl.style.color = '#fbbf24';
            else confidenceEl.style.color = '#f87171';
        }

        // Status indicator
        if (liveModeActive) {
            statusMsg.textContent = 'LIVE CAPTURE ACTIVE';
            statusMsg.style.color = '#67e8f9';
        } else if (data.running) {
            if (data.attack_active) {
                statusMsg.textContent = `INJECTING: ${data.attack_active}`;
                statusMsg.style.color = '#f87171';
            } else {
                statusMsg.textContent = 'Monitoring Active';
                statusMsg.style.color = '#34d399';
            }
        } else {
            statusMsg.textContent = 'System Standby';
            statusMsg.style.color = '#64748b';
        }

        // Charts
        liveTrafficChart.data.datasets[0].data = data.throughput;
        liveTrafficChart.update();

        // Class distribution from cumulative counts
        const counts = data.class_counts;
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        if (total === 0) counts['Normal'] = 1;
        threatPieChart.data.datasets[0].data = CLASS_LABELS.map(l => counts[l] || 0);
        threatPieChart.update();

        // Feed count
        feedCount.textContent = `${data.stats.attacks} events`;

        // Event Feed Table
        let feedHTML = '';
        data.alerts.forEach(alert => {
            const metaColor = CLASS_COLORS[alert.meta_class] || '#94a3b8';
            const ipClass = alert.severity === 'BLOCKED' ? 'ip-blocked' : 'ip-safe';
            const liveTag = alert.source === 'live' ? '<span style="color:#67e8f9;font-size:0.6rem;margin-left:4px">LIVE</span>' : '';
            feedHTML += `<tr>
                <td style="color:#64748b">${alert.time}</td>
                <td class="${ipClass}">${alert.ip}</td>
                <td><span style="color:${metaColor};font-weight:600">${alert.meta_class}</span>${liveTag}</td>
                <td>${alert.emoji} ${alert.type}</td>
                <td>${alert.confidence}%</td>
                <td><span class="table-badge ${alert.badge_class}">${alert.ips_action}</span></td>
            </tr>`;
        });
        feedBody.innerHTML = feedHTML;

        // IPS Prevention Actions
        if (data.ips.recent_actions.length > 0) {
            let ipsHTML = '';
            data.ips.recent_actions.slice().reverse().forEach(action => {
                const actionLower = action.action_taken.toLowerCase();
                let itemClass = 'action-alert';
                let badgeClass = 'alert';
                if (actionLower === 'blocked') { itemClass = 'action-blocked'; badgeClass = 'blocked'; }
                else if (actionLower === 'unblocked') { itemClass = 'action-unblocked'; badgeClass = 'alert'; }

                ipsHTML += `<div class="ips-action-item ${itemClass}">
                    <span class="action-badge ${badgeClass}">${action.action_taken}</span>
                    <span class="action-ip">${action.source_ip}</span>
                    <span class="action-type">${action.attack_type}</span>
                    <span class="action-time">${action.timestamp.split('T')[1]?.substring(0,8) || ''}</span>
                </div>`;
            });
            ipsActionsList.innerHTML = ipsHTML;
        }

        // Blocked IPs sidebar
        if (data.ips.blocked_ips.length > 0) {
            blockedIpsList.innerHTML = data.ips.blocked_ips
                .map(ip => `<span class="blocked-ip-tag">🔒 ${ip}</span>`)
                .join('');
        } else {
            blockedIpsList.innerHTML = '';
        }

        // Concept Drift Banner
        const banner = document.getElementById('driftBanner');
        if (data.drift) {
            if (data.drift.status === 'DRIFT_CRITICAL') {
                banner.style.display = 'flex';
                banner.style.borderLeftColor = '#f87171';
                document.getElementById('driftIcon').textContent = '🚨';
                document.getElementById('driftTitle').textContent = 'CRITICAL CONCEPT DRIFT';
                document.getElementById('driftTitle').style.color = '#f87171';
                document.getElementById('driftMessage').textContent = 'Major shift — model retraining recommended.';
                document.getElementById('driftSeverity').textContent = `PSI: ${data.drift.severity.toFixed(3)}`;
                document.getElementById('driftSeverity').style.color = '#f87171';
            } else if (data.drift.status === 'DRIFT_WARNING') {
                banner.style.display = 'flex';
                banner.style.borderLeftColor = '#fbbf24';
                document.getElementById('driftIcon').textContent = '⚠️';
                document.getElementById('driftTitle').textContent = 'Drift Warning';
                document.getElementById('driftTitle').style.color = '#fbbf24';
                document.getElementById('driftMessage').textContent = 'Minor distribution shift detected.';
                document.getElementById('driftSeverity').textContent = `PSI: ${data.drift.severity.toFixed(3)}`;
                document.getElementById('driftSeverity').style.color = '#fbbf24';
            } else {
                banner.style.display = 'none';
            }
        }

    } catch (e) {
        console.error('Connection lost:', e);
    }
}

// Poll every 500ms
setInterval(updateDashboard, 500);
