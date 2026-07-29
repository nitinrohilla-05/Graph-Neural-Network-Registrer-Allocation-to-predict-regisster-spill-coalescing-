// Interactive Graph Neural Network Register Allocation Dashboard Logic

let dataset = null;
let currentMode = "gnn"; // "gnn" or "cb"
let nodePositions = {};
let isDragging = false;
let draggedNode = null;
let hoverNode = null;

const colorMap = {
    "R0": "#3b82f6",
    "R1": "#10b981",
    "R2": "#f59e0b",
    "R3": "#ec4899",
    "SPILL": "#ef4444"
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    setupTabSwitching();
    setupModeToggle();
    setupCanvasInteractions();

    document.getElementById("btn-reload").addEventListener("click", loadData);
    loadData();
});

async function loadData() {
    try {
        const response = await fetch("data.json");
        if (!response.ok) throw new Error("Failed to fetch data.json");
        dataset = await response.json();
        renderDashboard();
    } catch (err) {
        console.warn("Could not fetch data.json, loading embedded fallback data:", err);
        dataset = getFallbackData();
        renderDashboard();
    }
}

function renderDashboard() {
    if (!dataset) return;

    // Stat cards
    document.getElementById("k-value").textContent = `K = ${dataset.num_registers}`;
    document.getElementById("stat-num-vars").textContent = dataset.nodes.length;
    document.getElementById("stat-interf-edges").textContent = dataset.interference_edges.length;
    document.getElementById("stat-coal-edges").textContent = dataset.coalescing_edges.length;
    document.getElementById("stat-gnn-spills").textContent = dataset.gnn_spills.length;
    document.getElementById("stat-cb-spills").textContent = dataset.cb_spills.length;

    // Render TAC IR
    const codeEl = document.getElementById("code-tac");
    codeEl.textContent = dataset.instructions.join("\n");

    // Render CFG Blocks
    renderCFG();

    // Render Register Table
    renderTable();

    // Initialize Graph Layout
    initForceLayout();
    drawGraph();
}

function setupTabSwitching() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const targetTab = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
            document.getElementById(targetTab).classList.add("active");
        });
    });
}

function setupModeToggle() {
    const gnnBtn = document.getElementById("mode-gnn");
    const cbBtn = document.getElementById("mode-cb");

    gnnBtn.addEventListener("click", () => {
        currentMode = "gnn";
        gnnBtn.classList.add("active");
        cbBtn.classList.remove("active");
        drawGraph();
    });

    cbBtn.addEventListener("click", () => {
        currentMode = "cb";
        cbBtn.classList.add("active");
        gnnBtn.classList.remove("active");
        drawGraph();
    });
}

function renderCFG() {
    const container = document.getElementById("cfg-blocks-container");
    container.innerHTML = "";

    dataset.cfg_blocks.forEach(block => {
        const card = document.createElement("div");
        card.className = "cfg-block-card";

        const succs = block.successors.length ? block.successors.join(", ") : "None (Exit)";
        card.innerHTML = `
            <div class="cfg-header">
                <span>Block ${block.block_id} ${block.label ? `(${block.label})` : ''}</span>
                <span>Loop Depth: ${block.loop_depth} | Succs: [${succs}]</span>
            </div>
            <pre class="code-block" style="padding: 0.5rem; margin-top: 0.4rem;"><code>${block.instructions.join("\n")}</code></pre>
        `;
        container.appendChild(card);
    });
}

function renderTable() {
    const body = document.getElementById("table-node-body");
    body.innerHTML = "";

    dataset.nodes.forEach(n => {
        const row = document.createElement("tr");

        const gnnReg = n.gnn_assignment;
        const cbReg = n.cb_assignment;

        const gnnBg = colorMap[gnnReg] || "#64748b";
        const cbBg = colorMap[cbReg] || "#64748b";

        row.innerHTML = `
            <td><strong>${n.id}</strong></td>
            <td>${n.loop_depth}</td>
            <td>${n.spill_cost.toFixed(1)}</td>
            <td>${n.degree}</td>
            <td><span class="badge-reg" style="background:${gnnBg}20; color:${gnnBg}; border:1px solid ${gnnBg}">${gnnReg}</span></td>
            <td><span class="badge-reg" style="background:${cbBg}20; color:${cbBg}; border:1px solid ${cbBg}">${cbReg}</span></td>
        `;
        body.appendChild(row);
    });
}

// Simple Spring Force Layout Engine
function initForceLayout() {
    const canvas = document.getElementById("graph-canvas");
    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;

    canvas.width = width;
    canvas.height = height;

    const N = dataset.nodes.length;
    const radius = Math.min(width, height) * 0.35;
    const centerX = width / 2;
    const centerY = height / 2;

    dataset.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / N;
        nodePositions[node.id] = {
            x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 40,
            y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 40,
            vx: 0,
            vy: 0
        };
    });

    // Run 50 iterations of force simulation
    for (let iter = 0; iter < 50; iter++) {
        // Repulsion
        for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
                const id1 = dataset.nodes[i].id;
                const id2 = dataset.nodes[j].id;
                const p1 = nodePositions[id1];
                const p2 = nodePositions[id2];

                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let dist = Math.sqrt(dx * dx + dy * dy) || 1;

                if (dist < 180) {
                    let force = (180 - dist) / dist * 0.2;
                    p1.x -= dx * force;
                    p1.y -= dy * force;
                    p2.x += dx * force;
                    p2.y += dy * force;
                }
            }
        }

        // Attraction over edges
        const allEdges = [...dataset.interference_edges, ...dataset.coalescing_edges];
        allEdges.forEach(edge => {
            const p1 = nodePositions[edge.source];
            const p2 = nodePositions[edge.target];
            if (!p1 || !p2) return;

            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let dist = Math.sqrt(dx * dx + dy * dy) || 1;

            let force = (dist - 120) * 0.03;
            p1.x += dx * force;
            p1.y += dy * force;
            p2.x -= dx * force;
            p2.y -= dy * force;
        });
    }
}

function drawGraph() {
    const canvas = document.getElementById("graph-canvas");
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw Interference Edges
    dataset.interference_edges.forEach(edge => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (!p1 || !p2) return;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.stroke();
    });

    // 2. Draw Coalescing Edges
    dataset.coalescing_edges.forEach(edge => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (!p1 || !p2) return;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
    });

    // 3. Draw Nodes
    dataset.nodes.forEach(node => {
        const pos = nodePositions[node.id];
        if (!pos) return;

        const assignment = (currentMode === "gnn") ? node.gnn_assignment : node.cb_assignment;
        const color = colorMap[assignment] || "#94a3b8";

        // Glow effect
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 22, 0, 2 * Math.PI);
        ctx.fillStyle = color + "33";
        ctx.fill();

        // Node Circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 16, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.lineWidth = hoverNode === node.id ? 3 : 1.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();

        // Label
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(node.id, pos.x, pos.y);
    });
}

function setupCanvasInteractions() {
    const canvas = document.getElementById("graph-canvas");
    const tooltip = document.getElementById("node-tooltip");

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        let found = null;
        dataset?.nodes.forEach(node => {
            const pos = nodePositions[node.id];
            if (!pos) return;
            const dist = Math.hypot(mouseX - pos.x, mouseY - pos.y);
            if (dist < 18) {
                found = node;
            }
        });

        if (found) {
            hoverNode = found.id;
            tooltip.classList.remove("hidden");
            tooltip.style.left = `${mouseX + 15}px`;
            tooltip.style.top = `${mouseY + 15}px`;
            tooltip.innerHTML = `
                <strong>Variable ${found.id}</strong><br/>
                Loop Depth: ${found.loop_depth}<br/>
                Spill Cost: ${found.spill_cost.toFixed(1)}<br/>
                GNN Reg: <span style="color:${colorMap[found.gnn_assignment]}">${found.gnn_assignment}</span><br/>
                Chaitin Reg: <span style="color:${colorMap[found.cb_assignment]}">${found.cb_assignment}</span>
            `;
        } else {
            hoverNode = null;
            tooltip.classList.add("hidden");
        }

        drawGraph();
    });
}

// Embedded Fallback Data if server/JSON loading fails
function getFallbackData() {
    return {
        "program_name": "func_demo",
        "num_registers": 4,
        "instructions": [
            "L00: v0 = const_1",
            "L01: v1 = const_5",
            "L02: v2 = v0 + v1",
            "L03: MOVE v3 <- v2",
            "L04: v4 = v3 * v1",
            "L05: RETURN v4"
        ],
        "cfg_blocks": [
            {
                "block_id": 0,
                "label": null,
                "loop_depth": 0,
                "instructions": ["v0 = const_1", "v1 = const_5", "v2 = v0 + v1", "MOVE v3 <- v2", "v4 = v3 * v1", "RETURN v4"],
                "successors": []
            }
        ],
        "nodes": [
            {"id": "v0", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 0, "gnn_assignment": "R0", "cb_assignment": "R0"},
            {"id": "v1", "spill_cost": 1.0, "loop_depth": 0, "degree": 2, "move_degree": 0, "gnn_assignment": "R1", "cb_assignment": "R1"},
            {"id": "v2", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 1, "gnn_assignment": "R2", "cb_assignment": "R2"},
            {"id": "v3", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 1, "gnn_assignment": "R2", "cb_assignment": "R2"},
            {"id": "v4", "spill_cost": 1.0, "loop_depth": 0, "degree": 0, "move_degree": 0, "gnn_assignment": "R0", "cb_assignment": "R0"}
        ],
        "interference_edges": [
            {"source": "v0", "target": "v1", "type": "interference"},
            {"source": "v1", "target": "v2", "type": "interference"}
        ],
        "coalescing_edges": [
            {"source": "v2", "target": "v3", "type": "coalescing"}
        ],
        "cb_coalesced_pairs": [["v2", "v3"]],
        "cb_spills": [],
        "gnn_spills": []
    };
}
