(() => {
  const $ = id => document.getElementById(id);
  const tabs = [...document.querySelectorAll('.tab')];
  const uploads = { image: $('upload-image'), files: $('upload-files'), video: $('upload-video'), text: $('upload-text') };

  const thread = $('thread');
  const threadEmpty = $('thread-empty');
  const questionInput = $('question-input');
  const sendBtn = $('send-btn');

  const COPY = {
    image: {
      empty: 'Upload an X-ray or photo, then ask about it.',
      ready: 'Ask anything about this image.',
      ask: { xray: ['What did the model find?', 'Are there any cavities?', 'How many teeth can you see?'],
             photo: ['Describe this image', 'Is there any text in it?'] },
    },
    files: { empty: 'Upload a document, then ask about it.', ready: 'Ask anything about this document.',
             ask: { document: ['Summarize this in 5 bullet points', 'What are the key dates?'] } },
    video: { empty: 'Load a video, then ask about it.', ready: 'Ask anything about this video.',
             ask: { video: ['Summarize this video', 'What are the main points?'] } },
    text: { ready: 'Ask about Near East University, or anything else.',
            ask: { knowledge: ['When was Near East University founded?', 'Who is the current rector?', 'What is fluoride varnish?'] } },
  };
  const SUB = { xray: 'Read by the dental model', photo: 'Described by GPT vision', document: 'Ready to ask', video: 'Transcript ready' };
  const SWATCH = { 'Fillings': 'fillings', 'Implant': 'implant', 'Cavity': 'cavity', 'Impacted Tooth': 'impacted' };

  // what's loaded in each tab: { name, kind, annotated, findings }
  const loaded = {};
  let mode = 'image';

  // ---------- layout ----------
  function render() {
    tabs.forEach(t => {
      const on = t.dataset.mode === mode;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', String(on));
    });
    const src = loaded[mode];
    Object.entries(uploads).forEach(([m, el]) => { el.hidden = m !== mode || (!!src && m !== 'text'); });

    $('result').hidden = !src;
    if (src) {
      $('result-name').textContent = src.name;
      $('result-sub').textContent = SUB[src.kind] || '';
      $('xray-result').hidden = !src.annotated;
      if (src.annotated) $('xray-result').src = src.annotated;
      renderFindings(src.findings);
    }

    const ready = mode === 'text' || !!src;
    questionInput.disabled = !ready;
    sendBtn.disabled = !ready;
    questionInput.placeholder = ready ? 'Ask a question…' : 'Load something first…';
    resetThread();
  }

  function renderFindings(findings) {
    const list = $('findings');
    list.replaceChildren();
    list.hidden = $('findings-note').hidden = !findings;
    if (!findings) return;
    findings.forEach(f => {
      const li = document.createElement('li');
      li.className = f.count ? '' : 'none';
      li.innerHTML = `<span class="swatch sw-${SWATCH[f.cls]}"></span><span class="f-name"></span><span class="f-count"></span><span class="f-where"></span>`;
      li.querySelector('.f-name').textContent = f.cls;
      li.querySelector('.f-count').textContent = f.count ? String(f.count) : 'none found';
      li.querySelector('.f-where').textContent = f.where;
      list.appendChild(li);
    });
  }

  function resetThread() {
    thread.querySelectorAll('.msg').forEach(el => el.remove());
    const src = loaded[mode];
    const copy = COPY[mode];
    $('thread-empty-text').textContent = mode === 'text' || src ? copy.ready : copy.empty;
    const suggestions = $('suggestions');
    suggestions.replaceChildren();
    const kind = mode === 'text' ? 'knowledge' : src && src.kind;
    (copy.ask[kind] || []).forEach(q => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'suggestion';
      b.textContent = q;
      b.addEventListener('click', () => { questionInput.value = q; $('composer').requestSubmit(); });
      suggestions.appendChild(b);
    });
    threadEmpty.hidden = false;
  }

  function setMode(m) {
    mode = m in uploads ? m : 'image';
    history.replaceState(null, '', '#' + mode);
    render();
  }
  tabs.forEach(t => t.addEventListener('click', () => setMode(t.dataset.mode)));

  // ---------- loading ----------
  async function loadSource(m, formData, btn) {
    formData.set('mode', m);
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = m === 'image' ? 'Analyzing…' : 'Reading…';
    try {
      const res = await fetch('/api/load', { method: 'POST', body: formData });
      const data = await res.json().catch(() => ({ error: 'The server sent an unexpected reply. Try again.' }));
      if (!res.ok) throw new Error(data.error || 'Couldn’t load that.');
      loaded[m] = { name: data.name, kind: data.kind, annotated: data.annotated, findings: data.findings };
      if (m === mode) { render(); questionInput.focus(); }
    } catch (err) {
      pushMessage('error', err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  }

  function forget(m) {
    delete loaded[m];
    if (m === 'image') clearPickedImage();
  }

  $('source-clear-btn').addEventListener('click', async () => {
    const m = mode;
    forget(m);
    render();
    await fetch('/api/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: m }) });
  });

  function wireDropzone(zone, onFile) {
    ['dragover', 'dragleave', 'drop'].forEach(evt => zone.addEventListener(evt, e => {
      e.preventDefault();
      zone.classList.toggle('drag', evt === 'dragover');
    }));
    zone.addEventListener('drop', e => { if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
  }

  // documents
  let pickedDocument = null;
  function pickDocument(file) {
    pickedDocument = file || null;
    $('files-picked-name').textContent = file ? file.name : '';
    $('files-load-btn').disabled = !file;
  }
  $('document-input').addEventListener('change', e => pickDocument(e.target.files[0]));
  wireDropzone($('dropzone-files'), pickDocument);
  $('files-load-btn').addEventListener('click', () => {
    if (!pickedDocument) return;
    const fd = new FormData();
    fd.set('document_file', pickedDocument);
    loadSource('files', fd, $('files-load-btn'));
  });

  // images
  let pickedImage = null;
  let cameraStream = null;
  function pickImage(file) {
    if (!file) return;
    pickedImage = file;
    $('image-picked-name').textContent = file.name;
    $('image-preview').src = URL.createObjectURL(file);
    $('image-preview-box').hidden = false;
    $('dropzone-image').hidden = true;
  }
  function clearPickedImage() {
    pickedImage = null;
    $('image-input').value = '';
    $('image-preview-box').hidden = true;
    $('dropzone-image').hidden = false;
  }
  $('image-input').addEventListener('change', e => pickImage(e.target.files[0]));
  wireDropzone($('dropzone-image'), pickImage);
  $('image-remove-btn').addEventListener('click', clearPickedImage);
  document.querySelectorAll('[data-sample]').forEach(btn => btn.addEventListener('click', async () => {
    const res = await fetch(btn.dataset.sample);
    const blob = await res.blob();
    pickImage(new File([blob], btn.dataset.sample.split('/').pop(), { type: blob.type }));
    $('image-load-btn').click();
  }));
  $('image-load-btn').addEventListener('click', () => {
    if (!pickedImage) return;
    const fd = new FormData();
    fd.set('image_file', pickedImage);
    loadSource('image', fd, $('image-load-btn'));
  });

  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    $('camera-live').hidden = true;
  }
  $('camera-open-btn').addEventListener('click', async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      $('camera-video').srcObject = cameraStream;
      $('camera-live').hidden = false;
    } catch (err) {
      pushMessage('error', 'Camera unavailable. Allow camera access in your browser, or upload a file instead.');
    }
  });
  $('camera-close-btn').addEventListener('click', stopCamera);
  $('camera-shoot-btn').addEventListener('click', () => {
    const video = $('camera-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      stopCamera();
      pickImage(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.92);
  });

  // video
  $('video-url-input').addEventListener('input', e => { $('video-load-btn').disabled = !e.target.value.trim(); });
  $('video-load-btn').addEventListener('click', () => {
    const fd = new FormData();
    fd.set('video_url', $('video-url-input').value.trim());
    loadSource('video', fd, $('video-load-btn'));
  });

  // ---------- conversation ----------
  // Small, safe subset of markdown: escape everything first, then add paragraphs, lists, headings and bold.
  function renderMarkdown(md) {
    const esc = md.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const inline = t => t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    const out = [];
    let list = null;
    for (const line of esc.split('\n')) {
      const item = line.match(/^\s*(?:[-*]|(\d+)\.)\s+(.*)/);
      if (item) {
        const tag = item[1] ? 'ol' : 'ul';
        if (list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; }
        out.push(`<li>${inline(item[2])}</li>`);
        continue;
      }
      if (list) { out.push(`</${list}>`); list = null; }
      const heading = line.match(/^#{1,6}\s+(.*)/);
      if (heading) out.push(`<h4>${inline(heading[1])}</h4>`);
      else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list}>`);
    return out.join('');
  }

  function scrollDown() { thread.scrollTop = thread.scrollHeight; }

  function pushMessage(kind, text) {
    threadEmpty.hidden = true;
    const el = document.createElement('div');
    el.className = `msg msg-${kind}`;
    el.textContent = text;
    thread.appendChild(el);
    scrollDown();
  }

  function pushAnswer(data) {
    threadEmpty.hidden = true;
    const el = document.createElement('div');
    el.className = `msg msg-assistant origin-${data.origin}`;
    const body = document.createElement('div');
    body.className = 'answer';
    body.innerHTML = renderMarkdown(data.answer);

    // where this answer came from
    const from = document.createElement('div');
    from.className = 'from';
    const tag = document.createElement('span');
    tag.className = 'origin';
    tag.textContent = 'From';  // the dot colour shows the kind of source
    from.appendChild(tag);
    (data.sources || []).forEach(s => {
      const chip = document.createElement(s.href ? 'a' : 'span');
      chip.className = 'loc';
      chip.textContent = s.label;
      if (s.href) { chip.href = s.href; chip.target = '_blank'; chip.rel = 'noopener'; }
      from.appendChild(chip);
    });

    el.append(body, from, makeSpeakButton(body));
    thread.appendChild(el);
    scrollDown();
  }

  function pushThinking() {
    threadEmpty.hidden = true;
    const el = document.createElement('div');
    el.className = 'msg msg-assistant thinking-wrap';
    el.setAttribute('aria-label', 'Thinking');
    el.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
    thread.appendChild(el);
    scrollDown();
    return el;
  }

  const synth = window.speechSynthesis;
  let speaking = null;
  function makeSpeakButton(bodyEl) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'speak-btn';
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4Z"/><path d="M17 8a5 5 0 0 1 0 8"/></svg> Listen';
    if (!synth) { btn.hidden = true; return btn; }
    btn.addEventListener('click', () => {
      if (speaking === btn) { synth.cancel(); return; }
      synth.cancel();
      const utter = new SpeechSynthesisUtterance(bodyEl.textContent);
      utter.onstart = () => { speaking = btn; btn.classList.add('playing'); };
      utter.onend = utter.onerror = () => { speaking = null; btn.classList.remove('playing'); };
      synth.speak(utter);
    });
    return btn;
  }

  $('composer').addEventListener('submit', async e => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question || questionInput.disabled) return;
    const askedMode = mode;

    pushMessage('user', question);
    questionInput.value = '';
    autoGrow();
    sendBtn.disabled = true;
    const thinking = pushThinking();

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: askedMode, question }),
      });
      const data = await res.json().catch(() => ({ error: 'The server sent an unexpected reply. Try again.' }));
      thinking.remove();
      if (data.reload) {  // server restarted or forgot the source: show the upload again
        forget(askedMode);
        if (askedMode === mode) render();
      }
      if (!res.ok) throw new Error(data.error || 'Couldn’t answer that. Try again.');
      if (askedMode === mode) pushAnswer(data);
    } catch (err) {
      thinking.remove();
      pushMessage('error', err.message);
    } finally {
      sendBtn.disabled = questionInput.disabled;
    }
  });

  function autoGrow() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
  }
  questionInput.addEventListener('input', autoGrow);
  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); }
  });

  // ---------- speech-to-text ----------
  const micBtn = $('mic-btn');
  const micStatus = $('mic-status');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || 'en-US';
    let finalTranscript = '';
    let listening = false;

    recognition.onstart = () => {
      finalTranscript = '';
      listening = true;
      micBtn.classList.add('listening');
      micStatus.textContent = 'Listening… press the microphone again to stop.';
    };
    recognition.onresult = event => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += (finalTranscript ? ' ' : '') + t;
        else interim += t;
      }
      micStatus.textContent = (finalTranscript + ' ' + interim).trim() || 'Listening…';
    };
    recognition.onend = () => {
      if (finalTranscript) {
        questionInput.value += (questionInput.value ? ' ' : '') + finalTranscript;
        autoGrow();
      }
      listening = false;
      micBtn.classList.remove('listening');
      micStatus.textContent = finalTranscript ? 'Check your question, then press Send.' : '';
    };
    recognition.onerror = event => {
      listening = false;
      micBtn.classList.remove('listening');
      micStatus.textContent = event.error === 'not-allowed'
        ? 'Microphone blocked. Allow it in your browser settings to ask out loud.'
        : 'Didn’t catch that. Try again.';
    };
    micBtn.addEventListener('click', () => {
      if (questionInput.disabled) { micStatus.textContent = 'Load something first, then ask out loud.'; return; }
      if (listening) recognition.stop(); else recognition.start();
    });
  } else {
    micBtn.disabled = true;
    micBtn.title = 'Asking out loud needs Chrome, Edge or Safari';
  }

  setMode(location.hash.slice(1));
})();
