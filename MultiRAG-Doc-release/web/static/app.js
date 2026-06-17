function app() {
  return {
    tab: 'query',
    debugMode: false,
    debugCollapsed: false,
    debugEvents: [],
    paperIds: [],
    papersFilter: '',
    modelStatus: { state: 'loading', message: '' },
    querySuggestions: [
      'What is the equation of the Attention?',
      'How does a Q-former works?',
      '什么是多头注意力?',
    ],
    agentSuggestions: [
      '这篇文章提出了什么创新点? 作者如何证明实验结果是有效的?',
      '找出支持论文主结论的图表，再解释这些证据之间的关系',
    ],
    modelingSuggestions: [
      'Build an optimization model for assigning jobs to machines to minimize processing cost and tardiness.',
      '为多订单多机器调度问题建立数学模型，目标是最小化加工成本和总延期。',
    ],

    q: {
      question: '',
      top_k: null,
      paper_id: '',
      generate_answer: false,
      stream: true,
      decompose: false,
      loading: false,
      text_results: [],
      figure_results: [],
      answer: null,
      guardrail_reason: '',
      streaming_text: '',
      plan: null,
    },

    ing: {
      file: null,
      paper_id: '',
      multimodal: false,
      use_caption_model: false,
      overwrite: false,
      loading: false,
      dragging: false,
      events: [],
      step_num: 0,
      result: null,
    },

    ag: {
      question: '',
      paper_id: '',
      max_steps: 10,
      loading: false,

      // Timeline 数据
      nodes: [],
      selected_node_id: null,
      auto_follow: true,

      // 状态
      status: 'idle',       // idle | running | done | aborted | error
      budget_used: 0,
      budget_cap: null,     // null 表示尚未收到 run_tool 事件，不渲染 Budget 条
      step_count: 0,
      error_message: '',

      // 答案区
      streaming_text: '',
      answer: null,
      warnings: [],
      guardrail_reason: '',
      terminate_reason: '',

      // 内部 SSE reader 引用（用于 abort）
      _reader: null,
    },

    mb: {
      problem: '',
      top_k: 6,
      paper_id: '',
      academic_mode: true,
      agent_mode: false,
      agent_max_rounds: 0,
      agent_quality_threshold: 0.86,
      loading: false,
      text_results: [],
      figure_results: [],
      model: null,
      modeling_plan: null,
      problem_spec: null,
      harness_draft: null,
      modeling_blueprint: null,
      model_verification: null,
      model_quality: null,
      code_generation: null,
      code_generation_error: '',
      agent_trace: [],
      agent_terminate_reason: '',
      skill: '',
      skill_description: '',
      generation_mode: '',
      raw_output: '',
      plan_output: '',
      plan_error: '',
      blueprint_output: '',
      blueprint_error: '',
      repair_output: '',
      revision_output: '',
      revision_note: '',
      parse_error: '',
      warnings: [],
      error_message: '',
      job_id: '',
      job_status: '',
      job_started_at: null,
      job_updated_at: null,
      job_elapsed_seconds: null,
      job_stage: '',
      job_message: '',
      job_progress: 0,
      platemo_root: 'PlatEMO',
      platemo_class_name: '',
      platemo_loading: false,
    },

    hhcEval: {
      top_k: 4,
      paper_id: '',
      full_llm: false,
      limit_cases: 1,
      case_ids_text: '',
      fast_llm_eval: true,
      save_report: true,
      loading: false,
      result: null,
      error_message: '',
      job_id: '',
      job_status: '',
      job_started_at: null,
      job_updated_at: null,
      job_elapsed_seconds: null,
      job_stage: '',
      job_message: '',
      job_progress: 0,
    },

    resultModal: {
      open: false,
      title: '',
      rank: null,
      result: null,
      renderedHtml: null,
      renderFailed: false,
    },

    imageModal: {
      open: false,
      title: '',
      index: null,
      zoom: 1,
      image: null,
    },

    async init() {
      await this.loadPapers();
      this._pollModelStatus();
    },

    async _pollModelStatus() {
      while (true) {
        try {
          const resp = await fetch('/api/status');
          const data = await resp.json();
          this.modelStatus = data;
          if (data.state === 'ready' || data.state === 'error') break;
        } catch (e) {
          // 服务尚未就绪，继续等待
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    },

    currentTabLabel() {
      return {
        papers: 'Papers',
        query: 'Query mode',
        ingest: 'Ingest mode',
        agent: 'Agent mode',
        modeling: 'Modeling mode',
      }[this.tab] || 'Workspace';
    },

    hasQueryOutput() {
      return Boolean(
        this.q.text_results.length ||
        this.q.figure_results.length ||
        this.q.answer ||
        this.q.streaming_text ||
        this.q.guardrail_reason ||
        (this.q.plan && this.q.plan.intent),
      );
    },

    showQueryIntro() {
      return !this.q.loading && !this.hasQueryOutput();
    },

    hasAgentOutput() {
      return Boolean(
        this.ag.nodes.length ||
        this.ag.streaming_text ||
        this.ag.answer ||
        this.ag.guardrail_reason,
      );
    },

    hasModelingOutput() {
      return Boolean(
        this.mb.model ||
        this.mb.modeling_plan ||
        this.mb.problem_spec ||
        this.mb.harness_draft ||
        this.mb.modeling_blueprint ||
        this.mb.raw_output ||
        this.mb.parse_error ||
        this.mb.error_message ||
        this.mb.agent_terminate_reason ||
        this.mb.agent_trace.length ||
        this.mb.code_generation ||
        this.mb.code_generation_error ||
        this.mb.text_results.length ||
        this.mb.figure_results.length,
      );
    },

    hasHhcEvalOutput() {
      return Boolean(
        this.hhcEval.loading ||
        this.hhcEval.result ||
        this.hhcEval.error_message,
      );
    },

    canRunModeling() {
      return !this.mb.loading && String(this.mb.problem || '').trim().length > 0;
    },

    canRunHhcEval() {
      return !this.hhcEval.loading && !this.mb.loading;
    },

    jobStatusLabel(status) {
      if (status === 'pending') return '排队中';
      if (status === 'running') return '运行中';
      if (status === 'cancelling') return '取消中';
      if (status === 'cancelled') return '已取消';
      if (status === 'done') return '已完成';
      if (status === 'error') return '出错';
      return status || '未开始';
    },

    jobProgressPercent(value) {
      const n = Number(value || 0);
      if (Number.isNaN(n)) return 0;
      return Math.max(0, Math.min(100, Math.round(n * 100)));
    },

    formatElapsedSeconds(value) {
      if (value == null || Number.isNaN(Number(value))) return '';
      const total = Math.max(0, Math.floor(Number(value)));
      const minutes = Math.floor(total / 60);
      const seconds = total % 60;
      if (!minutes) return `${seconds}s`;
      return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
    },

    hasOpenOverlay() {
      return this.resultModal.open || this.imageModal.open;
    },

    formatScore(score, digits = 3) {
      if (score == null || Number.isNaN(Number(score))) return '';
      return Number(score).toFixed(digits);
    },

    primaryResultScore(result) {
      if (!result) return null;
      if (result.modality !== 'figure' && result.rerank_score != null) {
        return Number(result.rerank_score);
      }
      if (result.score != null) {
        return Number(result.score);
      }
      return null;
    },

    normalizeAgentResults(payload) {
      if (payload && Array.isArray(payload.results) && payload.results.length) {
        return payload.results;
      }

      const textResults = Array.isArray(payload && payload.text_results) ? payload.text_results : [];
      const figureResults = Array.isArray(payload && payload.figure_results) ? payload.figure_results : [];
      return [...textResults, ...figureResults];
    },

    primaryResultScoreLabel(result) {
      if (!result) return 'score';
      if (result.modality !== 'figure' && result.rerank_score != null) {
        return 'rerank';
      }
      return 'score';
    },

    resultKey(result, fallback = '') {
      if (!result) return `result:${fallback}`;
      const stable = [
        result.chunk_id,
        result.figure_id,
        result.paper_id,
        result.page,
        result.modality,
        result.source_query,
        result.image_url,
        result.image_path,
        result.caption,
        result.content,
      ]
        .map((part) => String(part || '').trim())
        .filter((part) => part.length > 0)
        .join('|');
      return stable ? `result:${stable}` : `result:${fallback}`;
    },

    citationKey(citation, index) {
      if (!citation) return `citation:${index}`;
      const stable = [
        citation.chunk_id,
        citation.paper_id,
        citation.page,
        citation.modality,
        citation.content || citation.text || citation.excerpt,
      ]
        .map((part) => String(part || '').trim())
        .filter((part) => part.length > 0)
        .join('|');
      return stable ? `citation:${stable}` : `citation:${index}`;
    },

    sanitizeDomToken(value) {
      return String(value || '')
        .trim()
        .replace(/[^A-Za-z0-9_-]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'item';
    },

    evidenceDomId(scope, result, index = 0) {
      const key = [
        scope,
        result && result.modality,
        result && result.chunk_id,
        result && result.figure_id,
        result && result.paper_id,
        result && result.page,
        index,
      ]
        .map((part) => this.sanitizeDomToken(part))
        .join('-');
      return `evidence-${key}`;
    },

    evidenceCollection(scope) {
      if (scope === 'agent') {
        const items = [];
        for (const node of (this.ag && this.ag.nodes) || []) {
          if (node.type === 'run_tool' && node.data && node.data.new_items) {
            items.push(...node.data.new_items);
          }
        }
        return items;
      }
      if (scope === 'modeling') {
        return [
          ...(((this.mb && this.mb.text_results) || [])),
          ...(((this.mb && this.mb.figure_results) || [])),
        ];
      }
      return [
        ...(((this.q && this.q.text_results) || [])),
        ...(((this.q && this.q.figure_results) || [])),
      ];
    },

    findEvidenceMatch(scope, citation) {
      const evidence = this.evidenceCollection(scope);
      if (!citation || !evidence.length) return null;

      let matchedIndex = evidence.findIndex((item) => item.chunk_id && item.chunk_id === citation.chunk_id);
      if (matchedIndex >= 0) return { item: evidence[matchedIndex], index: matchedIndex };

      matchedIndex = evidence.findIndex(
        (item) => item.figure_id && citation.figure_id && item.figure_id === citation.figure_id,
      );
      if (matchedIndex >= 0) return { item: evidence[matchedIndex], index: matchedIndex };

      matchedIndex = evidence.findIndex(
        (item) =>
          item.paper_id === citation.paper_id
          && String(item.page || '') === String(citation.page || '')
          && item.modality === citation.modality,
      );
      if (matchedIndex >= 0) return { item: evidence[matchedIndex], index: matchedIndex };

      return null;
    },

    flashEvidenceCard(element) {
      if (!element) return;
      element.classList.remove('evidence-flash');
      void element.offsetWidth;
      element.classList.add('evidence-flash');
      window.setTimeout(() => {
        element.classList.remove('evidence-flash');
      }, 1800);
    },

    clearEvidenceHighlight(scope) {
      const selector = `[id^="evidence-${this.sanitizeDomToken(scope)}-"].evidence-highlighted`;
      document.querySelectorAll(selector).forEach((element) => {
        element.classList.remove('evidence-highlighted');
      });
    },

    highlightEvidenceCard(scope, element) {
      if (!element) return;
      this.clearEvidenceHighlight(scope);
      element.classList.add('evidence-highlighted');
      this.flashEvidenceCard(element);
    },

    jumpToEvidence(scope, citation) {
      const match = this.findEvidenceMatch(scope, citation);
      if (!match) return;

      const element = document.getElementById(this.evidenceDomId(scope, match.item, match.index));
      if (!element) return;

      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      window.setTimeout(() => {
        this.highlightEvidenceCard(scope, element);
      }, 180);
    },

    decorateAnswerText(answer) {
      const rawText = String((answer && answer.answer) || '');
      if (!rawText) return rawText;
      return rawText.replace(
        /\[(\d+)\]/g,
        '<span class="inline-citation" data-citation-index="$1">[$1]</span>',
      );
    },

    filteredPaperIds() {
      const keyword = this.papersFilter.trim().toLowerCase();
      if (!keyword) return this.paperIds;
      return this.paperIds.filter((pid) => String(pid || '').toLowerCase().includes(keyword));
    },

    openResultModal(result, options = {}) {
      this.resultModal.open = true;
      this.resultModal.title = options.title || 'Result Detail';
      this.resultModal.rank = options.rank ?? null;
      this.resultModal.result = result ? { ...result } : null;
      this.resultModal.renderedHtml = null;
      this.resultModal.renderFailed = false;

      if (result && result.modality === 'equation') {
        try {
          const raw = String(result.content || result.caption || '').trim();
          if (!raw) {
            this.resultModal.renderFailed = true;
            return;
          }

          if (typeof katex !== 'undefined') {
            const el = document.createElement('div');
            katex.render(raw, el, {
              displayMode: true,
              throwOnError: true,
            });
            this.resultModal.renderedHtml = el.innerHTML;
            return;
          }

          this.resultModal.renderFailed = true;
        } catch (_) {
          this.resultModal.renderFailed = true;
        }
      }
    },

    closeResultModal() {
      this.resultModal.open = false;
      this.resultModal.title = '';
      this.resultModal.rank = null;
      this.resultModal.result = null;
      this.resultModal.renderedHtml = null;
      this.resultModal.renderFailed = false;
    },

    openImageModal(image, options = {}) {
      this.imageModal.open = true;
      this.imageModal.title = options.title || 'Image Viewer';
      this.imageModal.index = options.index ?? null;
      this.imageModal.zoom = 1;
      this.imageModal.image = image ? { ...image } : null;
    },

    closeImageModal() {
      this.imageModal.open = false;
      this.imageModal.title = '';
      this.imageModal.index = null;
      this.imageModal.zoom = 1;
      this.imageModal.image = null;
    },

    closeOverlays() {
      this.closeResultModal();
      this.closeImageModal();
    },

    zoomImage(delta) {
      const next = this.imageModal.zoom + delta;
      this.imageModal.zoom = Math.min(4, Math.max(0.5, Number(next.toFixed(2))));
    },

    resetImageZoom() {
      this.imageModal.zoom = 1;
    },

    toggleImageZoom() {
      this.imageModal.zoom = this.imageModal.zoom > 1 ? 1 : 2;
    },

    setQuestion(question, target = 'query') {
      if (target === 'agent') {
        this.tab = 'agent';
        this.ag.question = question;
        return;
      }
      if (target === 'modeling') {
        this.tab = 'modeling';
        this.mb.problem = question;
        return;
      }
      this.tab = 'query';
      this.q.question = question;
    },

    toggleDebugMode() {
      this.debugMode = !this.debugMode;
      if (this.debugMode) {
        this.debugCollapsed = false;
      }
    },

    newQuery() {
      this.tab = 'query';
      this.q.question = '';
      this.q.top_k = null;
      this.q.paper_id = '';
      this.q.generate_answer = false;
      this.q.stream = true;
      this.q.decompose = false;
      this.q.loading = false;
      this.q.text_results = [];
      this.q.figure_results = [];
      this.q.answer = null;
      this.q.guardrail_reason = '';
      this.q.streaming_text = '';
      this.q.plan = null;
    },

    async loadPapers() {
      try {
        const resp = await fetch('/api/papers');
        const data = await resp.json();
        this.paperIds = (data.paper_ids || []).sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
      } catch (e) {
        console.error('loadPapers error:', e);
      }
    },

    async switchTab(t) {
      this.tab = t;
      if (t === 'papers' || t === 'query' || t === 'agent' || t === 'ingest' || t === 'modeling') {
        await this.loadPapers();
      }
    },

    async runQuery() {
      if (!this.q.question.trim() || this.q.loading) return;
      this.q.loading = true;
      this.q.text_results = [];
      this.q.figure_results = [];
      this.q.answer = null;
      this.q.guardrail_reason = '';
      this.q.streaming_text = '';
      this.q.plan = null;

      try {
        if (this.q.decompose) {
          if (this.q.generate_answer && this.q.stream) {
            await this._decomposeStream();
          } else {
            await this._decomposeNormal();
          }
        } else if (this.q.generate_answer && this.q.stream) {
          await this._queryStream();
        } else {
          await this._queryNormal();
        }
      } catch (e) {
        console.error('query error:', e);
      } finally {
        this.q.loading = false;
      }
    },

    async _queryNormal() {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.q.question,
          top_k: this.q.top_k,
          paper_id: this.q.paper_id || null,
          generate_answer: this.q.generate_answer,
        }),
      });
      const data = await resp.json();
      this.q.text_results = data.text_results || [];
      this.q.figure_results = data.figure_results || [];
      this.q.answer = data.answer;
      this.q.guardrail_reason = data.guardrail_reason || '';
    },

    async _queryStream() {
      const resp = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.q.question,
          top_k: this.q.top_k,
          paper_id: this.q.paper_id || null,
          generate_answer: true,
        }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === 'retrieval_done') {
            this.q.text_results = event.text_results || [];
            this.q.figure_results = event.figure_results || [];
          } else if (event.type === 'token') {
            if (typeof event.delta === 'string' && event.delta.length > 0) {
              this.q.streaming_text += event.delta;
            }
          } else if (event.type === 'done') {
            this.q.answer = event.answer;
            this.q.guardrail_reason = event.guardrail_reason || '';
            break;
          } else if (event.type === 'error') {
            console.error('stream error:', event.message);
            break;
          }
        }
      }
    },

    async _decomposeNormal() {
      const resp = await fetch('/api/query/decompose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.q.question,
          top_k: this.q.top_k || null,
          paper_id: this.q.paper_id || null,
          generate_answer: this.q.generate_answer,
        }),
      });
      const data = await resp.json();
      this.q.text_results = data.text_results || [];
      this.q.figure_results = data.figure_results || [];
      this.q.answer = data.answer;
      this.q.guardrail_reason = data.guardrail_reason || '';
      this.q.plan = (data.plan && data.plan.intent) ? data.plan : null;
    },

    async _decomposeStream() {
      const resp = await fetch('/api/query/decompose/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.q.question,
          top_k: this.q.top_k || null,
          paper_id: this.q.paper_id || null,
          generate_answer: true,
        }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === 'plan') {
            this.q.plan = (event.plan && event.plan.intent) ? event.plan : null;
          } else if (event.type === 'retrieval_done') {
            this.q.text_results = event.text_results || [];
            this.q.figure_results = event.figure_results || [];
          } else if (event.type === 'token') {
            if (typeof event.delta === 'string' && event.delta.length > 0) {
              this.q.streaming_text += event.delta;
            }
          } else if (event.type === 'done') {
            this.q.answer = event.answer;
            this.q.guardrail_reason = event.guardrail_reason || '';
            break;
          } else if (event.type === 'error') {
            console.error('decompose stream error:', event.message);
            break;
          }
        }
      }
    },

    handleFileDrop(e) {
      this.ing.dragging = false;
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.pdf')) {
        this.ing.file = file;
        if (!this.ing.paper_id) {
          this.ing.paper_id = file.name.replace(/\.pdf$/i, '').replace(/\s+/g, '_').replace(/[^A-Za-z0-9_-]/g, '');
        }
      }
    },

    handleFileSelect(e) {
      const file = e.target.files[0];
      if (file) {
        this.ing.file = file;
        if (!this.ing.paper_id) {
          this.ing.paper_id = file.name.replace(/\.pdf$/i, '').replace(/\s+/g, '_').replace(/[^A-Za-z0-9_-]/g, '');
        }
      }
    },

    async startIngest() {
      if (!this.ing.file || !this.ing.paper_id.trim() || this.ing.loading) return;
      this.ing.loading = true;
      this.ing.events = [];
      this.ing.step_num = 0;
      this.ing.result = null;

      const fd = new FormData();
      fd.append('file', this.ing.file);

      const params = new URLSearchParams({
        paper_id: this.ing.paper_id.trim(),
        multimodal: this.ing.multimodal,
        use_caption_model: this.ing.use_caption_model,
        overwrite: this.ing.overwrite,
      });

      try {
        const resp = await fetch(`/api/ingest?${params}`, { method: 'POST', body: fd });
        if (!resp.ok) {
          const err = await resp.json();
          const detail = Array.isArray(err.detail)
            ? err.detail.map(d => d.msg || JSON.stringify(d)).join('; ')
            : (err.detail || resp.statusText);
          this.ing.events.push({ type: 'error', _log: '[error] ' + detail });
          return;
        }
        const { job_id } = await resp.json();
        await this._watchIngest(job_id);
      } catch (e) {
        this.ing.events.push({ type: 'error', _log: '[error] ' + e.message });
      } finally {
        this.ing.loading = false;
      }
    },

    async _watchIngest(job_id) {
      const es = new EventSource(`/api/ingest/stream/${job_id}`);
      await new Promise((resolve) => {
        es.onmessage = (e) => {
          const event = JSON.parse(e.data);
          if (event.type === 'heartbeat') return;
          if (event.type === 'progress') {
            const stepNum = parseInt(event.step, 10);
            this.ing.step_num = Math.max(this.ing.step_num, stepNum || 0);
            event._log = `[${event.step}] ${event.message}`;
          } else if (event.type === 'done') {
            this.ing.step_num = 4;
            this.ing.result = event.result;
            event._log = '[done] 入库完成 ✓';
            this.loadPapers();
          } else if (event.type === 'error') {
            event._log = `[error] ${event.message}`;
          } else {
            event._log = JSON.stringify(event);
          }
          this.ing.events.push(event);
          if (event.type === 'done' || event.type === 'error') {
            es.close();
            resolve();
          }
        };
        es.onerror = () => {
          es.close();
          resolve();
        };
      });
    },

    pushDebugEvent(type, data, options = {}) {
      const now = new Date();
      const ts = now.toTimeString().slice(0, 8);
      const isPromptEvent = type === 'debug_llm' || type === 'answer_prompt';
      this.debugEvents.push({
        type,
        label: options.label || '',
        call_id: options.call_id || '',
        _ts: ts,
        _defaultOpen: options.defaultOpen ?? isPromptEvent,
        _data: data,
        ...(isPromptEvent ? { prompt: data.prompt, raw_output: data.raw_output } : {}),
      });
    },

    renderMarkdown(text) {
      if (!text) return '';
      const html = typeof marked === 'undefined' ? text : marked.parse(text);
      if (typeof renderMathInElement === 'undefined') return html;

      try {
        const container = document.createElement('div');
        container.innerHTML = html;
        renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
          ],
          throwOnError: false,
        });
        return container.innerHTML;
      } catch (_) {
        return html;
      }
    },

    renderAnswer(answer) {
      if (!answer || !answer.answer) return '';
      return this.renderMarkdown(this.decorateAnswerText(answer));
    },

    async copyAnswerAsMarkdown(scope) {
      const answer = scope === 'agent' ? this.ag.answer : this.q.answer;
      if (!answer || !answer.answer) return;

      let md = answer.answer;

      if (answer.citations && answer.citations.length) {
        md += '\n\n---\n**Citations:**\n';
        answer.citations.forEach((c, i) => {
          const idx = c.index || (i + 1);
          const paper = c.paper_id || 'unknown';
          const page = c.page ? ` p.${c.page}` : '';
          const content = c.content || c.text || c.excerpt || '';
          md += `\n[${idx}] ${paper}${page}: ${content.slice(0, 200)}${content.length > 200 ? '...' : ''}\n`;
        });
      }

      try {
        await navigator.clipboard.writeText(md);
      } catch (e) {
        console.error('Copy failed:', e);
      }
    },

    modelingEvidence() {
      return [
        ...((this.mb && this.mb.text_results) || []),
        ...((this.mb && this.mb.figure_results) || []),
      ];
    },

    modelEvidenceTags(item) {
      if (!item) return [];
      const tags = [];
      if (item.chunk_type) tags.push(item.chunk_type);
      const elements = Array.isArray(item.model_elements) ? item.model_elements : [];
      elements.forEach((value) => tags.push(value));
      const operators = Array.isArray(item.operator_hints) ? item.operator_hints : [];
      operators.slice(0, 3).forEach((value) => tags.push(value));
      const hhc = Array.isArray(item.hhc_signals) ? item.hhc_signals : [];
      hhc.slice(0, 3).forEach((value) => tags.push(`hhc:${value}`));
      return [...new Set(tags.filter(Boolean))].slice(0, 8);
    },

    modelFieldList(field) {
      const value = this.mb && this.mb.model ? this.mb.model[field] : [];
      return Array.isArray(value) ? value : [];
    },

    componentApplicabilityList() {
      const value = this.mb && this.mb.model ? this.mb.model.component_applicability : [];
      return Array.isArray(value) ? value : [];
    },

    modelingPlanList(field) {
      const plan = this.mb && this.mb.modeling_plan ? this.mb.modeling_plan : {};
      const value = plan ? plan[field] : [];
      return Array.isArray(value) ? value : [];
    },

    problemSpec() {
      return (this.mb && this.mb.problem_spec) || {};
    },

    problemSpecList(field) {
      const spec = this.problemSpec();
      const value = spec ? spec[field] : [];
      return Array.isArray(value) ? value : [];
    },

    problemSpecSymbols(field) {
      const spec = this.problemSpec();
      const symbols = spec && spec.recommended_symbols ? spec.recommended_symbols : {};
      const value = symbols ? symbols[field] : [];
      return Array.isArray(value) ? value : [];
    },

    modelingBlueprintList(field) {
      const blueprint = this.mb && this.mb.modeling_blueprint ? this.mb.modeling_blueprint : {};
      const value = blueprint ? blueprint[field] : [];
      return Array.isArray(value) ? value : [];
    },

    harnessSpecList(field) {
      const spec = this.mb && this.mb.harness_draft && this.mb.harness_draft.model_spec
        ? this.mb.harness_draft.model_spec
        : {};
      const value = spec ? spec[field] : [];
      return Array.isArray(value) ? value : [];
    },

    harnessComponentSelector() {
      const value = this.mb && this.mb.harness_draft ? this.mb.harness_draft.component_selector : [];
      return Array.isArray(value) ? value : [];
    },

    componentStatusClass(status) {
      if (status === 'selected') return 'border border-emerald-200 bg-emerald-50 text-emerald-700';
      if (status === 'omitted') return 'border border-stone-200 bg-stone-100 text-stone-600';
      return 'border border-amber-200 bg-amber-50 text-amber-700';
    },

    componentStatusLabel(status) {
      if (status === 'selected') return 'Selected';
      if (status === 'omitted') return 'Omitted';
      return 'Not selected';
    },

    harnessSymbolList(field) {
      const plan = this.mb && this.mb.harness_draft && this.mb.harness_draft.symbol_plan
        ? this.mb.harness_draft.symbol_plan
        : {};
      const value = plan ? plan[field] : [];
      return Array.isArray(value) ? value : [];
    },

    harnessValidation() {
      return (this.mb && this.mb.harness_draft && this.mb.harness_draft.validation) || {};
    },

    sourceLabel(ids) {
      if (!Array.isArray(ids) || !ids.length) return 'no direct source';
      return ids.join(', ');
    },

    renderFormula(value) {
      const text = String(value || '').trim();
      if (!text) return '';
      const escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      const html = `<div>${escaped}</div>`;
      if (typeof renderMathInElement === 'undefined') return html;

      try {
        const container = document.createElement('div');
        container.innerHTML = html;
        renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
        return container.innerHTML;
      } catch (_) {
        return html;
      }
    },

    objectiveFormula() {
      const obj = this.mb && this.mb.model && this.mb.model.objective;
      return obj && obj.formula ? obj.formula : '';
    },

    validationList(field) {
      const validation = this.mb && this.mb.model && this.mb.model.validation;
      const value = validation ? validation[field] : [];
      return Array.isArray(value) ? value : [];
    },

    hhcEvalSummary() {
      return (this.hhcEval && this.hhcEval.result && this.hhcEval.result.summary) || {};
    },

    hhcEvalRecords() {
      const records = this.hhcEval && this.hhcEval.result ? this.hhcEval.result.records : [];
      return Array.isArray(records) ? records : [];
    },

    hhcEvalStatusClass(status) {
      if (status === 'PASS') return 'border border-emerald-200 bg-emerald-50 text-emerald-700';
      if (status === 'WARN') return 'border border-amber-200 bg-amber-50 text-amber-700';
      if (status === 'FAIL') return 'border border-red-200 bg-red-50 text-red-700';
      if (status === 'ERROR') return 'border border-sky-200 bg-sky-50 text-sky-700';
      return 'border border-stone-200 bg-stone-100 text-stone-600';
    },

    hhcEvalOutputPath() {
      return this.hhcEval && this.hhcEval.result ? (this.hhcEval.result.output_path || '') : '';
    },

    hhcEvalCaseIds() {
      const raw = String(this.hhcEval && this.hhcEval.case_ids_text || '');
      return raw
        .split(/[,\n]/)
        .map((value) => value.trim())
        .filter((value) => value.length > 0);
    },

    updateHhcEvalJob(job) {
      const now = Date.now() / 1000;
      this.hhcEval.job_id = job.job_id || this.hhcEval.job_id || '';
      this.hhcEval.job_status = job.status || '';
      this.hhcEval.job_started_at = job.created_at || this.hhcEval.job_started_at;
      this.hhcEval.job_updated_at = job.updated_at || null;
      this.hhcEval.job_stage = job.stage || '';
      this.hhcEval.job_message = job.message || '';
      this.hhcEval.job_progress = Number(job.progress || 0);
      const end = job.terminal_at || now;
      if (this.hhcEval.job_started_at) {
        this.hhcEval.job_elapsed_seconds = Math.max(0, end - this.hhcEval.job_started_at);
      }
    },

    async pollHhcEvalJob(jobId) {
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const resp = await fetch(`/api/modeling/eval/hhc/jobs/${encodeURIComponent(jobId)}`);
        if (!resp.ok) {
          let detail = resp.statusText || 'HHC eval job status request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const job = await resp.json();
        this.updateHhcEvalJob(job);
        if (job.status === 'done') {
          this.hhcEval.result = job.result || null;
          return;
        }
        if (job.status === 'error') {
          throw new Error(job.error || 'HHC eval job failed');
        }
        if (job.status === 'cancelled') {
          throw new Error('任务已取消');
        }
      }
    },

    async cancelHhcEvalJob() {
      if (!this.hhcEval.job_id || !this.hhcEval.loading) return;
      try {
        const resp = await fetch(`/api/modeling/eval/hhc/jobs/${encodeURIComponent(this.hhcEval.job_id)}/cancel`, {
          method: 'POST',
        });
        if (resp.ok) {
          const job = await resp.json();
          this.updateHhcEvalJob(job);
        }
      } catch (e) {
        console.error('cancel hhc eval job failed:', e);
      }
    },

    loadHhcEvalRecord(record) {
      if (!record) return;
      this.mb.problem = record.problem || '';
      this.mb.top_k = this.hhcEval.top_k || this.mb.top_k || 4;
      this.mb.paper_id = this.hhcEval.paper_id || '';
      this.hhcEval.case_ids_text = record.id || '';
      requestAnimationFrame(() => {
        const el = document.querySelector('[data-modeling-composer]');
        if (el && typeof el.scrollIntoView === 'function') {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    },

    async replayHhcEvalRecord(record, mode = 'harness') {
      this.loadHhcEvalRecord(record);
      await new Promise((resolve) => setTimeout(resolve, 120));
      if (mode === 'full') {
        await this.runModeling();
      } else {
        await this.runHarnessFormula();
      }
    },

    modelVerification() {
      return (this.mb && this.mb.model_verification) || null;
    },

    modelQuality() {
      return (this.mb && this.mb.model_quality) || null;
    },

    modelQualityScores() {
      const quality = this.modelQuality();
      const scores = quality && quality.scores ? quality.scores : {};
      return Object.entries(scores).map(([name, score]) => ({ name, score }));
    },

    modelingAgentTrace() {
      const trace = this.mb && this.mb.agent_trace ? this.mb.agent_trace : [];
      return Array.isArray(trace) ? trace : [];
    },

    platemoCodeGeneration() {
      return (this.mb && this.mb.code_generation) || null;
    },

    verificationChecks() {
      const verification = this.modelVerification();
      const value = verification ? verification.checks : [];
      return Array.isArray(value) ? value : [];
    },

    verificationStatusClass(status) {
      if (status === 'pass') return 'border border-emerald-200 bg-emerald-50 text-emerald-700';
      if (status === 'warn') return 'border border-amber-200 bg-amber-50 text-amber-700';
      if (status === 'fail') return 'border border-red-200 bg-red-50 text-red-700';
      return 'border border-stone-200 bg-stone-100 text-stone-600';
    },

    verificationStatusLabel(status) {
      if (status === 'pass') return 'Pass';
      if (status === 'warn') return 'Warn';
      if (status === 'fail') return 'Fail';
      return 'Unknown';
    },

    resetModelingOutput() {
      this.mb.text_results = [];
      this.mb.figure_results = [];
      this.mb.model = null;
      this.mb.modeling_plan = null;
      this.mb.problem_spec = null;
      this.mb.harness_draft = null;
      this.mb.modeling_blueprint = null;
      this.mb.model_verification = null;
      this.mb.model_quality = null;
      this.mb.code_generation = null;
      this.mb.code_generation_error = '';
      this.mb.agent_trace = [];
      this.mb.agent_terminate_reason = '';
      this.mb.skill = '';
      this.mb.skill_description = '';
      this.mb.generation_mode = '';
      this.mb.raw_output = '';
      this.mb.plan_output = '';
      this.mb.plan_error = '';
      this.mb.blueprint_output = '';
      this.mb.blueprint_error = '';
      this.mb.repair_output = '';
      this.mb.revision_output = '';
      this.mb.revision_note = '';
      this.mb.parse_error = '';
      this.mb.warnings = [];
      this.mb.error_message = '';
      this.mb.job_id = '';
      this.mb.job_status = '';
      this.mb.job_started_at = null;
      this.mb.job_updated_at = null;
      this.mb.job_elapsed_seconds = null;
      this.mb.job_stage = '';
      this.mb.job_message = '';
      this.mb.job_progress = 0;
      this.mb.platemo_loading = false;
      this.mb.platemo_class_name = '';
    },

    applyModelingResult(data) {
      this.mb.text_results = data.text_results || [];
      this.mb.figure_results = data.figure_results || [];
      this.mb.model = data.model || null;
      this.mb.modeling_plan = data.modeling_plan || null;
      this.mb.problem_spec = data.problem_spec || null;
      this.mb.harness_draft = data.harness_draft || null;
      this.mb.modeling_blueprint = data.modeling_blueprint || null;
      this.mb.model_verification = data.model_verification || null;
      this.mb.model_quality = data.model_quality || null;
      this.mb.code_generation = data.code_generation || null;
      this.mb.code_generation_error = '';
      this.mb.agent_trace = data.agent_trace || [];
      this.mb.agent_terminate_reason = data.agent_terminate_reason || '';
      this.mb.skill = data.skill || '';
      this.mb.skill_description = data.skill_description || '';
      this.mb.generation_mode = data.generation_mode || '';
      this.mb.raw_output = data.raw_output || '';
      this.mb.plan_output = data.plan_output || '';
      this.mb.plan_error = data.plan_error || '';
      this.mb.blueprint_output = data.blueprint_output || '';
      this.mb.blueprint_error = data.blueprint_error || '';
      this.mb.repair_output = data.repair_output || '';
      this.mb.revision_output = data.revision_output || '';
      this.mb.revision_note = data.revision_note || '';
      this.mb.parse_error = data.parse_error || '';
      this.mb.warnings = data.warnings || [];
    },

    updateModelingJob(job) {
      const now = Date.now() / 1000;
      this.mb.job_id = job.job_id || this.mb.job_id || '';
      this.mb.job_status = job.status || '';
      this.mb.job_started_at = job.created_at || this.mb.job_started_at;
      this.mb.job_updated_at = job.updated_at || null;
      this.mb.job_stage = job.stage || '';
      this.mb.job_message = job.message || '';
      this.mb.job_progress = Number(job.progress || 0);
      const end = job.terminal_at || now;
      if (this.mb.job_started_at) {
        this.mb.job_elapsed_seconds = Math.max(0, end - this.mb.job_started_at);
      }
    },

    async pollModelingJob(jobId) {
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const resp = await fetch(`/api/modeling/generate/jobs/${encodeURIComponent(jobId)}`);
        if (!resp.ok) {
          let detail = resp.statusText || 'Modeling job status request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const job = await resp.json();
        this.updateModelingJob(job);
        if (job.status === 'done') {
          this.applyModelingResult(job.result || {});
          return;
        }
        if (job.status === 'error') {
          throw new Error(job.error || 'Modeling job failed');
        }
        if (job.status === 'cancelled') {
          throw new Error('任务已取消');
        }
      }
    },

    async cancelModelingJob() {
      if (!this.mb.job_id || !this.mb.loading) return;
      try {
        const resp = await fetch(`/api/modeling/generate/jobs/${encodeURIComponent(this.mb.job_id)}/cancel`, {
          method: 'POST',
        });
        if (resp.ok) {
          const job = await resp.json();
          this.updateModelingJob(job);
        }
      } catch (e) {
        console.error('cancel modeling job failed:', e);
      }
    },

    async runModeling(options = {}) {
      if (!this.canRunModeling()) return;
      const agentMode = Boolean(options.agentMode);
      this.mb.loading = true;
      this.resetModelingOutput();
      try {
        const resp = await fetch('/api/modeling/generate/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            problem: this.mb.problem,
            top_k: this.mb.top_k || null,
            paper_id: this.mb.paper_id || null,
            academic_mode: Boolean(this.mb.academic_mode),
            agent_mode: agentMode,
            agent_max_rounds: agentMode ? (this.mb.agent_max_rounds ?? 0) : null,
            agent_quality_threshold: agentMode ? (this.mb.agent_quality_threshold || 0.86) : null,
          }),
        });
        if (!resp.ok) {
          let detail = resp.statusText || 'Modeling request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const job = await resp.json();
        this.updateModelingJob(job);
        await this.pollModelingJob(job.job_id);
      } catch (e) {
        console.error('modeling error:', e);
        this.mb.error_message = e.message || String(e);
        if (!this.mb.job_status || this.mb.loading) this.mb.job_status = 'error';
      } finally {
        this.mb.loading = false;
      }
    },

    async runHarnessDraft() {
      if (!this.canRunModeling()) return;
      this.mb.loading = true;
      this.resetModelingOutput();

      try {
        const resp = await fetch('/api/modeling/harness', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            problem: this.mb.problem,
            top_k: this.mb.top_k || null,
            paper_id: this.mb.paper_id || null,
          }),
        });
        if (!resp.ok) {
          let detail = resp.statusText || 'Harness request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const data = await resp.json();
        this.mb.text_results = data.text_results || [];
        this.mb.figure_results = data.figure_results || [];
        this.mb.model = data.model || null;
        this.mb.harness_draft = data.harness_draft || null;
        this.mb.model_verification = data.model_verification || null;
        this.mb.model_quality = data.model_quality || null;
        this.mb.skill = data.skill || '';
        this.mb.skill_description = data.skill_description || '';
        this.mb.generation_mode = data.generation_mode || '';
        this.mb.warnings = data.warnings || [];
      } catch (e) {
        console.error('harness error:', e);
        this.mb.error_message = e.message || String(e);
      } finally {
        this.mb.loading = false;
      }
    },

    async runHarnessFormula() {
      if (!this.canRunModeling()) return;
      this.mb.loading = true;
      this.resetModelingOutput();

      try {
        const resp = await fetch('/api/modeling/harness', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            problem: this.mb.problem,
            top_k: this.mb.top_k || null,
            paper_id: this.mb.paper_id || null,
            render_formulas: true,
          }),
        });
        if (!resp.ok) {
          let detail = resp.statusText || 'Harness formula request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const data = await resp.json();
        this.mb.text_results = data.text_results || [];
        this.mb.figure_results = data.figure_results || [];
        this.mb.model = data.model || null;
        this.mb.harness_draft = data.harness_draft || null;
        this.mb.model_verification = data.model_verification || null;
        this.mb.model_quality = data.model_quality || null;
        this.mb.skill = data.skill || '';
        this.mb.skill_description = data.skill_description || '';
        this.mb.generation_mode = data.generation_mode || '';
        this.mb.warnings = data.warnings || [];
      } catch (e) {
        console.error('harness formula error:', e);
        this.mb.error_message = e.message || String(e);
      } finally {
        this.mb.loading = false;
      }
    },

    async runHhcEval() {
      if (!this.canRunHhcEval()) return;
      this.hhcEval.loading = true;
      this.hhcEval.result = null;
      this.hhcEval.error_message = '';
      this.hhcEval.job_id = '';
      this.hhcEval.job_status = '';
      this.hhcEval.job_started_at = null;
      this.hhcEval.job_updated_at = null;
      this.hhcEval.job_elapsed_seconds = null;
      this.hhcEval.job_stage = '';
      this.hhcEval.job_message = '';
      this.hhcEval.job_progress = 0;

      try {
        const resp = await fetch('/api/modeling/eval/hhc/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            top_k: this.hhcEval.top_k || 4,
            paper_id: this.hhcEval.paper_id || null,
            full_llm: Boolean(this.hhcEval.full_llm),
            limit_cases: this.hhcEval.full_llm ? (this.hhcEval.limit_cases || 1) : null,
            case_ids: this.hhcEvalCaseIds(),
            fast_llm_eval: Boolean(this.hhcEval.fast_llm_eval),
            save_report: Boolean(this.hhcEval.save_report),
          }),
        });
        if (!resp.ok) {
          let detail = resp.statusText || 'HHC eval request failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const job = await resp.json();
        this.updateHhcEvalJob(job);
        await this.pollHhcEvalJob(job.job_id);
      } catch (e) {
        console.error('hhc eval error:', e);
        this.hhcEval.error_message = e.message || String(e);
        if (!this.hhcEval.job_status || this.hhcEval.loading) this.hhcEval.job_status = 'error';
      } finally {
        this.hhcEval.loading = false;
      }
    },

    async copyModelJson() {
      if (!this.mb.model) return;
      try {
        await navigator.clipboard.writeText(JSON.stringify(this.mb.model, null, 2));
      } catch (e) {
        console.error('Copy model JSON failed:', e);
      }
    },

    async runPlatemoCodegen() {
      if (!this.mb.model && !this.mb.harness_draft) return;
      this.mb.platemo_loading = true;
      this.mb.code_generation_error = '';
      try {
        const resp = await fetch('/api/modeling/platemo-code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            problem: this.mb.problem || '',
            model: this.mb.model || null,
            harness_draft: this.mb.harness_draft || null,
            problem_spec: this.mb.problem_spec || null,
            platemo_root: this.mb.platemo_root || null,
            class_name: this.mb.platemo_class_name || null,
            write_file: true,
          }),
        });
        if (!resp.ok) {
          let detail = resp.statusText || 'PlatEMO code generation failed';
          try {
            const err = await resp.json();
            detail = err.detail || err.message || detail;
          } catch (_) {
            try { detail = await resp.text() || detail; } catch (_) {}
          }
          throw new Error(detail);
        }
        const data = await resp.json();
        this.mb.code_generation = data;
        if (!this.mb.platemo_class_name && data.class_name) {
          this.mb.platemo_class_name = data.class_name;
        }
      } catch (e) {
        console.error('platemo code generation error:', e);
        this.mb.code_generation_error = e.message || String(e);
      } finally {
        this.mb.platemo_loading = false;
      }
    },

    async copyPlatemoCode() {
      const code = this.platemoCodeGeneration();
      if (!code || !code.matlab_code) return;
      try {
        await navigator.clipboard.writeText(code.matlab_code);
      } catch (e) {
        console.error('Copy PlatEMO code failed:', e);
      }
    },

    openCitation(c, ci) {
      const citationIndex = c && c.index ? c.index : (ci + 1);
      const result = {
        modality: c.modality || 'text',
        paper_id: c.paper_id || '',
        page: c.page || null,
        content: c.content || c.text || c.excerpt || '',
        score: c.score ?? null,
        rerank_score: c.rerank_score ?? null,
      };
      this.openResultModal(result, { title: `Citation [${citationIndex}]`, rank: citationIndex });
    },

    // ── Agent Timeline 辅助方法 ───────────────────────────────────────────────

    agSelectedNode() {
      if (!this.ag.selected_node_id) return null;
      return this.ag.nodes.find((n) => n.id === this.ag.selected_node_id) || null;
    },

    selectNode(id) {
      this.ag.selected_node_id = id;
      this.ag.auto_follow = false;
    },

    agNodeLabel(node) {
      return { plan_query: 'Plan Query', decide_action: 'Decide Action', run_tool: 'Run Tool', compress_evidence: 'Compress Evidence', answer_start: 'Answer' }[node.type] || node.type;
    },

    agNodeShortLabel(node) {
      return { plan_query: 'Plan', decide_action: 'Decide', run_tool: 'Tool', compress_evidence: 'Compress', answer_start: 'Ans' }[node.type] || node.type;
    },

    agNodeIcon(node) {
      if (node.status === 'running') {
        return '<svg class="spinner h-3.5 w-3.5 text-[var(--accent)]" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
      }
      if (node.status === 'done') {
        return '<svg class="h-3.5 w-3.5 text-[var(--accent-deep)]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" /></svg>';
      }
      if (node.status === 'error') {
        return '<svg class="h-3.5 w-3.5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>';
      }
      return '<svg class="h-3.5 w-3.5 text-[var(--line-strong)]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="5" stroke-width="2" /></svg>';
    },

    abortAgent() {
      if (this.ag._reader) {
        try { this.ag._reader.cancel(); } catch (_) {}
        this.ag._reader = null;
      }
      this.ag.status = 'aborted';
      this.ag.loading = false;
      this.ag.nodes.forEach((n) => { if (n.status === 'running') n.status = 'done'; });
    },

    async runAgent() {
      if (!this.ag.question.trim() || this.ag.status === 'running') return;

      // 重置所有状态
      this.ag.status = 'running';
      this.ag.loading = true;
      this.ag.nodes = [];
      this.ag.selected_node_id = null;
      this.ag.auto_follow = true;
      this.ag.budget_used = 0;
      this.ag.budget_cap = null;
      this.ag.step_count = 0;
      this.ag.error_message = '';
      this.ag.streaming_text = '';
      this.ag.answer = null;
      this.ag.warnings = [];
      this.ag.guardrail_reason = '';
      this.ag.terminate_reason = '';
      this.ag._reader = null;

      try {
        await this._agentStream();
      } catch (e) {
        if (this.ag.status !== 'aborted') {
          console.error('agent error:', e);
          this.ag.status = 'error';
          this.ag.error_message = e.message || String(e);
        }
      } finally {
        this.ag.loading = false;
        if (this.ag.status === 'running') this.ag.status = 'done';
      }
    },

    async _agentStream() {
      if (this.debugMode) this.debugEvents = [];
      const resp = await fetch('/api/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.ag.question,
          paper_id: this.ag.paper_id || null,
          generate_answer: true,
          max_steps: this.ag.max_steps,
          debug: this.debugMode,
        }),
      });
      if (!resp.ok) {
        let detail = resp.statusText || 'Agent request failed';
        try {
          const err = await resp.json();
          detail = err.detail || err.message || detail;
        } catch (_) {
          try {
            detail = await resp.text() || detail;
          } catch (_) {}
        }
        throw new Error(detail);
      }
      if (!resp.body) {
        throw new Error('Agent stream is unavailable');
      }

      const reader = resp.body.getReader();
      this.ag._reader = reader;
      const decoder = new TextDecoder();
      let buf = '';

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            let event;
            try { event = JSON.parse(line.slice(6)); } catch (_) { continue; }

            switch (event.type) {
              case 'plan_query': {
                const node = {
                  id: 'plan_query',
                  type: 'plan_query',
                  step: 0,
                  status: 'done',
                  indented: false,
                  subtitle: `${(event.sub_queries || []).length} sub-queries`,
                  data: event,
                };
                this.ag.nodes.push(node);
                if (this.ag.auto_follow) this.ag.selected_node_id = 'plan_query';
                if (this.debugMode) this.pushDebugEvent('plan_query', event, { label: `Plan: ${(event.sub_queries || []).length} sub-queries` });
                break;
              }
              case 'decide_action': {
                const step = event.step ?? 0;
                this.ag.step_count = Math.max(this.ag.step_count, step + 1);
                // Step 组头
                const groupId = `step_group_${step}`;
                if (!this.ag.nodes.find((n) => n.id === groupId)) {
                  this.ag.nodes.push({ id: groupId, type: 'step_group', step, status: 'done', indented: false, subtitle: '', data: {} });
                }
                const nodeId = `decide_${step}`;
                let subtitle = event.action || '';
                if (event.action === 'search_evidence' && event.args && event.args.query) {
                  subtitle = `"${String(event.args.query).slice(0, 30)}"`;
                }
                const existing = this.ag.nodes.findIndex((n) => n.id === nodeId);
                const node = { id: nodeId, type: 'decide_action', step, status: 'done', indented: true, subtitle, data: event };
                if (existing >= 0) this.ag.nodes.splice(existing, 1, node);
                else this.ag.nodes.push(node);
                if (this.ag.auto_follow) this.ag.selected_node_id = nodeId;
                if (this.debugMode) this.pushDebugEvent('decide_action', event, { label: `Step ${step}: ${event.action}${event.reasoning ? ' — ' + String(event.reasoning).slice(0, 40) : ''}` });
                break;
              }
              case 'run_tool': {
                const step = event.step ?? 0;
                this.ag.budget_used = event.total_evidence || 0;
                if (event.cap != null) this.ag.budget_cap = event.cap;
                const nodeId = `run_tool_${step}`;
                const subtitle = `+${event.new_evidence_count} evidence · ${(event.papers_hit || []).length} papers`;
                const existing = this.ag.nodes.findIndex((n) => n.id === nodeId);
                const node = { id: nodeId, type: 'run_tool', step, status: 'done', indented: true, subtitle, data: event };
                if (existing >= 0) this.ag.nodes.splice(existing, 1, node);
                else this.ag.nodes.push(node);
                if (this.ag.auto_follow) this.ag.selected_node_id = nodeId;
                if (this.debugMode) this.pushDebugEvent('run_tool', event, { label: `Step ${step}: +${event.new_evidence_count} evidence, budget ${event.total_evidence}/${event.cap}` });
                break;
              }
              case 'compress_evidence': {
                const step = event.step ?? 0;
                this.ag.budget_used = event.kept || 0;
                const nodeId = `compress_${step}`;
                const discarded = event.discarded || [];
                const subtitle = `${event.before} → ${event.kept} chunks (−${discarded.length})`;
                const node = { id: nodeId, type: 'compress_evidence', step, status: 'done', indented: true, subtitle, data: event };
                const existing = this.ag.nodes.findIndex((n) => n.id === nodeId);
                if (existing >= 0) this.ag.nodes.splice(existing, 1, node);
                else this.ag.nodes.push(node);
                if (this.ag.auto_follow) this.ag.selected_node_id = nodeId;
                if (this.debugMode) this.pushDebugEvent('compress_evidence', event, { label: `Step ${step}: compress ${event.before}→${event.kept} chunks` });
                break;
              }
              case 'answer_start': {
                this.ag.warnings = event.warnings || [];
                const subtitle = `${event.selected_evidence_count} evidence selected`;
                const node = { id: 'answer_start', type: 'answer_start', step: this.ag.step_count, status: 'running', indented: false, subtitle, data: event };
                const existing = this.ag.nodes.findIndex((n) => n.id === 'answer_start');
                if (existing >= 0) this.ag.nodes.splice(existing, 1, node);
                else this.ag.nodes.push(node);
                if (this.ag.auto_follow) this.ag.selected_node_id = 'answer_start';
                if (this.debugMode) this.pushDebugEvent('answer_start', event, { label: `Answer: ${event.selected_evidence_count} evidence, terminate=${event.terminate_reason}` });
                break;
              }
              case 'debug_llm': {
                const isAnswer = event.call_id === 'answer_prompt';
                this.pushDebugEvent(isAnswer ? 'answer_prompt' : 'debug_llm', event, {
                  call_id: event.call_id || '',
                  label: isAnswer
                    ? `Step ${event.step ?? '?'}: answer prompt`
                    : `Step ${event.step ?? '?'}: policy LLM`,
                  defaultOpen: true,
                });
                break;
              }
              case 'token': {
                if (typeof event.delta === 'string') this.ag.streaming_text += event.delta;
                break;
              }
              case 'done': {
                this.ag.status = 'done';
                this.ag.answer = event.answer;
                this.ag.guardrail_reason = event.guardrail_reason || '';
                this.ag.terminate_reason = event.terminate_reason || '';
                const ansNode = this.ag.nodes.find((n) => n.id === 'answer_start');
                if (ansNode) ansNode.status = 'done';
                return;
              }
              case 'error': {
                this.ag.status = 'error';
                this.ag.error_message = event.message || 'Unknown error';
                return;
              }
            }
          }
        }
      } finally {
        this.ag._reader = null;
      }
    },
  };
}
