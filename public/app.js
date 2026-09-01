/**
 * VerdictNet / Texas Legal Graph — Frontend Application Controller
 * Powered by Google Gemini 3.7 Flash & Neo4j Citation Networks
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- State Variables ---
    let currentStrategy = "defense";
    let networkInstance = null;
    let lastAnalysisData = null;
    let exampleCases = [];
    let isPhysicsEnabled = true;

    // --- DOM Elements ---
    const caseInput = document.getElementById("case-facts-input");
    const charCount = document.getElementById("char-count");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnClear = document.getElementById("btn-clear");
    const btnStratDefense = document.getElementById("btn-strat-defense");
    const btnStratProsecution = document.getElementById("btn-strat-prosecution");
    const analyzeSpinner = document.getElementById("analyze-spinner");
    const loadingPanel = document.getElementById("loading-panel");
    const loadingStepText = document.getElementById("loading-step-text");
    const resultsWrapper = document.getElementById("results-wrapper");
    const emptyState = document.getElementById("empty-state");
    const tierIndicator = document.getElementById("tier-indicator");
    const tierDescription = document.getElementById("tier-description");
    const resultsCountPill = document.getElementById("results-count-pill");
    const memoContent = document.getElementById("memo-content");
    const precedentsList = document.getElementById("precedents-list");
    const precedentsCounter = document.getElementById("precedents-counter");
    const networkContainer = document.getElementById("citation-network");
    const btnFitGraph = document.getElementById("btn-fit-graph");
    const btnTogglePhysics = document.getElementById("btn-toggle-physics");
    const btnCopyMemo = document.getElementById("btn-copy-memo");
    const btnPrintMemo = document.getElementById("btn-print-memo");
    const toast = document.getElementById("toast");
    const toastMessage = document.getElementById("toast-message");
    const dbStatusPill = document.getElementById("db-status-pill");
    const dbStatusText = document.getElementById("db-status-text");

    // Modal elements
    const opinionModal = document.getElementById("opinion-modal");
    const modalCaseTitle = document.getElementById("modal-case-title");
    const modalCaseMeta = document.getElementById("modal-case-meta");
    const modalCaseText = document.getElementById("modal-case-text");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnCloseModalBottom = document.getElementById("btn-modal-close-bottom");
    const btnModalCopy = document.getElementById("btn-modal-copy");

    // --- 1. Initial Health Check & Examples Loading ---
    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                const dot = dbStatusPill.querySelector(".status-dot");
                if (data.neo4j_connected) {
                    dot.classList.remove("offline");
                    dot.classList.add("online");
                    dbStatusText.textContent = "Neo4j Connected";
                } else if (data.demo_dataset_loaded) {
                    dot.classList.remove("offline");
                    dot.classList.add("online");
                    dbStatusText.textContent = `Graph Engine (${data.demo_dataset_loaded} Cases)`;
                } else {
                    dot.classList.remove("online");
                    dot.classList.add("offline");
                    dbStatusText.textContent = "Offline";
                }
            }
        } catch (e) {
            console.warn("Health check unreachable:", e);
        }
    }

    async function loadExamples() {
        try {
            const res = await fetch("/api/examples");
            if (res.ok) {
                exampleCases = await res.json();
                bindExampleChips();
            }
        } catch (e) {
            console.warn("Could not load examples:", e);
        }
    }

    function bindExampleChips() {
        const chipButtons = document.querySelectorAll(".chip-btn");
        chipButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const exId = btn.dataset.example;
                const found = exampleCases.find(c => c.id === exId);
                if (found) {
                    caseInput.value = found.facts;
                    updateCharCount();
                    setStrategy(found.recommended_strategy || "defense");
                    showToast(`Loaded: ${found.title}`);
                }
            });
        });
    }

    // --- 2. Strategy Selector Logic ---
    function setStrategy(strategy) {
        currentStrategy = strategy;
        if (strategy === "defense") {
            btnStratDefense.classList.add("active");
            btnStratProsecution.classList.remove("active");
        } else {
            btnStratProsecution.classList.add("active");
            btnStratDefense.classList.remove("active");
        }
    }

    btnStratDefense.addEventListener("click", () => setStrategy("defense"));
    btnStratProsecution.addEventListener("click", () => setStrategy("prosecution"));

    // --- 3. Input & Character Counter ---
    function updateCharCount() {
        const len = caseInput.value.trim().length;
        charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
    }

    caseInput.addEventListener("input", updateCharCount);

    btnClear.addEventListener("click", () => {
        caseInput.value = "";
        updateCharCount();
        resultsWrapper.style.display = "none";
        emptyState.style.display = "none";
        lastAnalysisData = null;
        showToast("Form cleared");
    });

    // --- 4. Toast Notification Utility ---
    let toastTimeout = null;
    function showToast(msg, duration = 3000) {
        if (toastTimeout) clearTimeout(toastTimeout);
        toastMessage.textContent = msg;
        toast.style.display = "block";
        toastTimeout = setTimeout(() => {
            toast.style.display = "none";
        }, duration);
    }

    // --- 5. Analyze Jurisprudence Pipeline ---
    btnAnalyze.addEventListener("click", handleAnalyze);

    async function handleAnalyze() {
        const text = caseInput.value.trim();
        if (!text || text.length < 5) {
            showToast("Please describe the case facts (at least 5 characters).");
            caseInput.focus();
            return;
        }

        // UI Loading State
        btnAnalyze.disabled = true;
        analyzeSpinner.style.display = "inline-block";
        loadingPanel.style.display = "block";
        resultsWrapper.style.display = "none";
        emptyState.style.display = "none";

        // Simulated progress steps
        const stepMessages = [
            "Generating query vector with text-embedding-004...",
            "Traversing Neo4j 2-hop citation network...",
            "Filtering precedents for strategic alignment...",
            "Synthesizing legal memorandum via Gemini 3.7 Flash..."
        ];
        let stepIdx = 0;
        loadingStepText.textContent = stepMessages[0];
        const stepInterval = setInterval(() => {
            stepIdx = (stepIdx + 1) % stepMessages.length;
            loadingStepText.textContent = stepMessages[stepIdx];
        }, 1200);

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    strategy: currentStrategy,
                    top_k: 5
                })
            });

            clearInterval(stepInterval);

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "Analysis request failed.");
            }

            const data = await res.json();
            lastAnalysisData = data;
            renderResults(data);

        } catch (err) {
            clearInterval(stepInterval);
            console.error("Analysis error:", err);
            showToast(`Error: ${err.message}`, 4500);
        } finally {
            btnAnalyze.disabled = false;
            analyzeSpinner.style.display = "none";
            loadingPanel.style.display = "none";
        }
    }

    // --- 6. Rendering Results ---
    function renderResults(data) {
        if (!data.precedents || data.precedents.length === 0) {
            resultsWrapper.style.display = "none";
            emptyState.style.display = "block";
            return;
        }

        emptyState.style.display = "none";
        resultsWrapper.style.display = "block";

        // Update Method Banner
        const method = data.method || "Gold (Filtered Graph RAG)";
        if (method.includes("Gold")) {
            tierIndicator.textContent = "Gold Standard";
            tierIndicator.style.background = "#FEF3C7";
            tierIndicator.style.color = "#92400E";
            tierDescription.textContent = "Retrieved via citation network traversal matching target strategy";
        } else if (method.includes("Silver")) {
            tierIndicator.textContent = "Silver Standard";
            tierIndicator.style.background = "#E0E7FF";
            tierIndicator.style.color = "#3730A3";
            tierDescription.textContent = "Retrieved via citation network traversal (unfiltered fallback)";
        } else {
            tierIndicator.textContent = "Bronze Baseline";
            tierIndicator.style.background = "#FEE2E2";
            tierIndicator.style.color = "#991B1B";
            tierDescription.textContent = "Selected based primarily on semantic vector similarity";
        }

        resultsCountPill.textContent = `${data.count} Precedent${data.count === 1 ? '' : 's'} Located`;
        precedentsCounter.textContent = `Showing ${data.count} Cases`;

        // Render Markdown Legal Memo
        if (typeof marked !== "undefined" && data.analysis) {
            memoContent.innerHTML = marked.parse(data.analysis);
        } else {
            memoContent.innerText = data.analysis || "No memo generated.";
        }

        // Render Precedents List
        renderPrecedentsList(data.precedents);

        // Render Vis.js Citation Graph
        renderCitationGraph(data.graph);

        // Scroll smoothly to results
        resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderPrecedentsList(precedents) {
        precedentsList.innerHTML = "";
        precedents.forEach((caseItem, idx) => {
            const card = document.createElement("div");
            const isFav = caseItem.is_favorable;
            const favorClass = isFav ? "favorable" : "adverse";
            card.className = `prec-card ${favorClass}`;
            card.id = `prec-card-${caseItem.id}`;

            const excerpt = caseItem.full_text 
                ? caseItem.full_text.slice(0, 160) + "..."
                : "No opinion excerpt available.";

            card.innerHTML = `
                <div class="prec-card-header">
                    <div class="prec-title">${escapeHtml(caseItem.title)}</div>
                    <span class="prec-match-pill">${caseItem.confidence_pct}% Match</span>
                </div>
                <div class="prec-meta">
                    <span class="decision-tag">${escapeHtml(caseItem.decision)}</span>
                    <span>${caseItem.citation_count} Citation${caseItem.citation_count === 1 ? '' : 's'}</span>
                    <span>Offense: ${escapeHtml(caseItem.offense)}</span>
                </div>
                <div class="prec-excerpt">"${escapeHtml(excerpt)}"</div>
                <button class="btn-read-opinion" data-case-index="${idx}">Read Full Opinion &rarr;</button>
            `;

            const readBtn = card.querySelector(".btn-read-opinion");
            readBtn.addEventListener("click", () => openOpinionModal(caseItem));

            precedentsList.appendChild(card);
        });
    }

    // --- 7. Vis.js Graph Rendering ---
    function renderCitationGraph(graphData) {
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            networkContainer.innerHTML = "<p style='padding:2rem;text-align:center;'>No citation graph data.</p>";
            return;
        }

        const nodes = new vis.DataSet(graphData.nodes);
        const edges = new vis.DataSet(graphData.edges);

        const data = { nodes, edges };

        const options = {
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 3,
                font: {
                    face: "Lato",
                    size: 12,
                    color: "#2C2520"
                },
                shadow: {
                    enabled: true,
                    color: "rgba(0,0,0,0.1)",
                    size: 5,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                smooth: {
                    type: "curvedCW",
                    roundness: 0.15
                },
                font: {
                    size: 9,
                    color: "#8B7D6B"
                }
            },
            physics: {
                enabled: isPhysicsEnabled,
                solver: "forceAtlas2Based",
                forceAtlas2Based: {
                    gravitationalConstant: -35,
                    centralGravity: 0.008,
                    springLength: 100,
                    springConstant: 0.08
                },
                stabilization: {
                    iterations: 150
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }
        };

        if (networkInstance) {
            networkInstance.destroy();
        }

        networkInstance = new vis.Network(networkContainer, data, options);

        // Interaction: Click Node to Highlight Precedent Card
        networkInstance.on("click", (params) => {
            if (params.nodes.length > 0) {
                const clickedNodeId = params.nodes[0];
                highlightPrecedentCard(clickedNodeId);
            }
        });
    }

    function highlightPrecedentCard(nodeId) {
        document.querySelectorAll(".prec-card").forEach(c => c.classList.remove("highlighted"));
        const targetCard = document.getElementById(`prec-card-${nodeId}`);
        if (targetCard) {
            targetCard.classList.add("highlighted");
            targetCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }

    // Vis.js graph controls
    btnFitGraph.addEventListener("click", () => {
        if (networkInstance) networkInstance.fit({ animation: { duration: 600 } });
    });

    btnTogglePhysics.addEventListener("click", () => {
        if (networkInstance) {
            isPhysicsEnabled = !isPhysicsEnabled;
            networkInstance.setOptions({ physics: { enabled: isPhysicsEnabled } });
            btnTogglePhysics.textContent = isPhysicsEnabled ? "Physics ON" : "Physics OFF";
            showToast(isPhysicsEnabled ? "Graph physics enabled" : "Graph physics frozen");
        }
    });

    // --- 8. Opinion Modal Operations ---
    function openOpinionModal(caseItem) {
        modalCaseTitle.textContent = caseItem.title || "Case Law Record";
        modalCaseMeta.textContent = `Offense: ${caseItem.offense || 'N/A'} • Decision: ${caseItem.decision || 'N/A'} • Authority Citations: ${caseItem.citation_count || 0}`;
        modalCaseText.textContent = caseItem.full_text || "No full judicial opinion text available.";
        opinionModal.style.display = "flex";
    }

    function closeOpinionModal() {
        opinionModal.style.display = "none";
    }

    btnCloseModal.addEventListener("click", closeOpinionModal);
    btnCloseModalBottom.addEventListener("click", closeOpinionModal);
    opinionModal.addEventListener("click", (e) => {
        if (e.target === opinionModal) closeOpinionModal();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && opinionModal.style.display === "flex") {
            closeOpinionModal();
        }
    });

    btnModalCopy.addEventListener("click", () => {
        navigator.clipboard.writeText(modalCaseText.textContent).then(() => {
            showToast("Opinion text copied to clipboard!");
        });
    });

    // --- 9. Memo Actions (Copy & Print) ---
    btnCopyMemo.addEventListener("click", () => {
        if (lastAnalysisData && lastAnalysisData.analysis) {
            navigator.clipboard.writeText(lastAnalysisData.analysis).then(() => {
                showToast("Strategic Memo copied to clipboard!");
            });
        }
    });

    btnPrintMemo.addEventListener("click", () => {
        window.print();
    });

    // --- Helper Utilities ---
    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // --- Run Initializers ---
    checkHealth();
    loadExamples();
    updateCharCount();
});
