/* =============================================================
   Kalkulator Metode Numerik — Logika Perhitungan
   Metode: Matriks Balikan | Lelaran Jacobi | Lelaran Gauss-Seidel
   ============================================================= */

(function(){
  "use strict";

  /* ===================== STATE ===================== */
  const state = { method: 'inverse', n: 2 };

  const els = {
    methodList: document.getElementById('methodList'),
    sizeVal: document.getElementById('sizeVal'),
    sizeUp: document.getElementById('sizeUp'),
    sizeDown: document.getElementById('sizeDown'),
    matrixA: document.getElementById('matrixA'),
    vectorB: document.getElementById('vectorB'),
    paramsBox: document.getElementById('paramsBox'),
    x0Row: document.getElementById('x0Row'),
    tolInput: document.getElementById('tolInput'),
    maxIterInput: document.getElementById('maxIterInput'),
    warnDD: document.getElementById('warnDD'),
    errorArea: document.getElementById('errorArea'),
    emptyState: document.getElementById('emptyState'),
    resultsContent: document.getElementById('resultsContent'),
    methodHint: document.getElementById('methodHint'),
    btnHitung: document.getElementById('btnHitung'),
    btnContoh: document.getElementById('btnContoh'),
    btnReset: document.getElementById('btnReset'),
    themeToggle: document.getElementById('themeToggle'),
    themeIcon: document.getElementById('themeIcon'),
    themeLabel: document.getElementById('themeLabel'),
  };

  const VARNAMES = ['x','y','z','w','v'];
  const METHOD_HINTS = {
    inverse: 'Menyelesaikan Ax = B melalui invers matriks A. Cocok untuk sistem berukuran kecil–menengah dan menghasilkan solusi eksak (langsung, tanpa iterasi).',
    jacobi: 'Memperbarui semua variabel secara serentak menggunakan nilai iterasi sebelumnya. Membutuhkan matriks dominan diagonal agar konvergen.',
    seidel: 'Penyempurnaan Jacobi: langsung memakai nilai terbaru dalam iterasi yang sama, sehingga umumnya konvergen lebih cepat.'
  };

  const EXAMPLES = {
    inverse: { A:[[3,1],[2,2]], B:[5,6] },
    jacobi:  { A:[[5,-2],[3,10]], B:[3,7], x0:[0,0], tol:0.0001, maxIter:50 },
    seidel:  { A:[[5,-2],[3,10]], B:[3,7], x0:[0,0], tol:0.0001, maxIter:50 }
  };

  /* ===================== GRID BUILDING ===================== */
  function buildGrids(){
    const n = state.n;
    // matrix A
    let html = '';
    for(let i=0;i<n;i++){
      html += '<tr>';
      for(let j=0;j<n;j++){
        html += `<td><input class="cell-input" type="text" inputmode="decimal" data-a-i="${i}" data-a-j="${j}" placeholder="0"></td>`;
      }
      html += '</tr>';
    }
    els.matrixA.innerHTML = html;

    // vector B
    let hb = '';
    for(let i=0;i<n;i++){
      hb += `<tr><td><input class="cell-input vec-input" type="text" inputmode="decimal" data-b-i="${i}" placeholder="0"></td></tr>`;
    }
    els.vectorB.innerHTML = hb;

    // x0 row (only meaningful for iterative, but build always so data is ready)
    let hx = '';
    for(let i=0;i<n;i++){
      hx += `<input type="text" inputmode="decimal" data-x0-i="${i}" placeholder="${VARNAMES[i]||'v'+i}₀ = 0" value="0">`;
    }
    els.x0Row.innerHTML = hx;

    clearResults();
    checkDiagonalDominanceLive();
  }

  function readNumberField(el){
    const raw = (el.value || '').trim().replace(',', '.');
    if(raw === '') return 0;
    const v = Number(raw);
    return isNaN(v) ? NaN : v;
  }

  function readMatrixA(){
    const n = state.n, A = [];
    for(let i=0;i<n;i++){
      A.push([]);
      for(let j=0;j<n;j++){
        const el = els.matrixA.querySelector(`[data-a-i="${i}"][data-a-j="${j}"]`);
        A[i].push(readNumberField(el));
      }
    }
    return A;
  }
  function readVectorB(){
    const n = state.n, B = [];
    for(let i=0;i<n;i++){
      const el = els.vectorB.querySelector(`[data-b-i="${i}"]`);
      B.push(readNumberField(el));
    }
    return B;
  }
  function readX0(){
    const n = state.n, x0 = [];
    for(let i=0;i<n;i++){
      const el = els.x0Row.querySelector(`[data-x0-i="${i}"]`);
      x0.push(readNumberField(el));
    }
    return x0;
  }

  function hasNaN(arr2dOr1d){
    return arr2dOr1d.some(v => Array.isArray(v) ? v.some(x=>isNaN(x)) : isNaN(v));
  }

  /* ===================== NUMERICAL CORE ===================== */

  // Gauss-Jordan elimination with partial pivoting.
  // Returns { inv, det, singular }
  function invertAndDet(Ain){
    const n = Ain.length;
    const M = Ain.map(r => r.slice());
    const I = Array.from({length:n}, (_,i)=> Array.from({length:n}, (_,j)=> i===j?1:0));
    let det = 1;

    for(let col=0; col<n; col++){
      // partial pivot: find max abs in this column at/below 'col'
      let pivotRow = col;
      let maxAbs = Math.abs(M[col][col]);
      for(let r=col+1;r<n;r++){
        if(Math.abs(M[r][col]) > maxAbs){ maxAbs = Math.abs(M[r][col]); pivotRow = r; }
      }
      if(maxAbs < 1e-12){
        return { inv:null, det:0, singular:true };
      }
      if(pivotRow !== col){
        [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
        [I[col], I[pivotRow]] = [I[pivotRow], I[col]];
        det *= -1;
      }
      const pivotVal = M[col][col];
      det *= pivotVal;
      for(let j=0;j<n;j++){ M[col][j] /= pivotVal; I[col][j] /= pivotVal; }

      for(let r=0;r<n;r++){
        if(r===col) continue;
        const factor = M[r][col];
        if(factor === 0) continue;
        for(let j=0;j<n;j++){
          M[r][j] -= factor * M[col][j];
          I[r][j] -= factor * I[col][j];
        }
      }
    }
    return { inv:I, det:det, singular:false };
  }

  function matVec(M, v){
    return M.map(row => row.reduce((s,val,j)=> s + val*v[j], 0));
  }

  function residualNorm(A,B,x){
    // ||Ax - B||_inf  -> validasi akurasi solusi langsung
    const Ax = matVec(A,x);
    let m = 0;
    for(let i=0;i<Ax.length;i++) m = Math.max(m, Math.abs(Ax[i]-B[i]));
    return m;
  }

  function checkDiagonalDominance(A){
    const n = A.length;
    let allWeak = true, strictAny = false;
    const rows = [];
    for(let i=0;i<n;i++){
      const diag = Math.abs(A[i][i]);
      let offSum = 0;
      for(let j=0;j<n;j++) if(j!==i) offSum += Math.abs(A[i][j]);
      const ok = diag >= offSum - 1e-12;
      const strict = diag > offSum + 1e-12;
      if(!ok) allWeak = false;
      if(strict) strictAny = true;
      rows.push({i, diag, offSum, ok});
    }
    return { dominant: allWeak && strictAny, rows };
  }

  function jacobi(A,B,x0,tol,maxIter){
    const n = A.length;
    let x = x0.slice();
    const history = [];
    let converged = false, diverged = false, iterUsed = 0;

    for(let it=1; it<=maxIter; it++){
      const xNew = new Array(n).fill(0);
      for(let i=0;i<n;i++){
        if(Math.abs(A[i][i]) < 1e-12){ diverged = true; break; }
        let sum = B[i];
        for(let j=0;j<n;j++){ if(j!==i) sum -= A[i][j]*x[j]; }
        xNew[i] = sum / A[i][i];
      }
      if(diverged) break;
      let errAbs = 0;
      for(let i=0;i<n;i++) errAbs = Math.max(errAbs, Math.abs(xNew[i]-x[i]));
      const denom = Math.max(...xNew.map(v=>Math.abs(v)), 1e-12);
      const errRel = errAbs/denom;

      history.push({ iter: it, x: xNew.slice(), errAbs, errRel });
      x = xNew;
      iterUsed = it;

      if(!isFinite(errAbs) || errAbs > 1e10 || x.some(v=>!isFinite(v))){ diverged = true; break; }
      if(errAbs < tol){ converged = true; break; }
    }
    return { x, history, converged, diverged, iterUsed };
  }

  function gaussSeidel(A,B,x0,tol,maxIter){
    const n = A.length;
    let x = x0.slice();
    const history = [];
    let converged = false, diverged = false, iterUsed = 0;

    for(let it=1; it<=maxIter; it++){
      const xOld = x.slice();
      for(let i=0;i<n;i++){
        if(Math.abs(A[i][i]) < 1e-12){ diverged = true; break; }
        let sum = B[i];
        for(let j=0;j<n;j++){ if(j!==i) sum -= A[i][j]*x[j]; }
        x[i] = sum / A[i][i];
      }
      if(diverged) break;
      let errAbs = 0;
      for(let i=0;i<n;i++) errAbs = Math.max(errAbs, Math.abs(x[i]-xOld[i]));
      const denom = Math.max(...x.map(v=>Math.abs(v)), 1e-12);
      const errRel = errAbs/denom;

      history.push({ iter: it, x: x.slice(), errAbs, errRel });
      iterUsed = it;

      if(!isFinite(errAbs) || errAbs > 1e10 || x.some(v=>!isFinite(v))){ diverged = true; break; }
      if(errAbs < tol){ converged = true; break; }
    }
    return { x, history, converged, diverged, iterUsed };
  }

  /* ===================== UI: rendering results ===================== */

  function clearResults(){
    els.emptyState.style.display = 'block';
    els.resultsContent.style.display = 'none';
    els.resultsContent.innerHTML = '';
    els.errorArea.innerHTML = '';
  }

  function showError(msg){
    els.errorArea.innerHTML = `<div class="error-banner">⚠ ${msg}</div>`;
  }

  function checkDiagonalDominanceLive(){
    if(state.method === 'inverse'){ els.warnDD.innerHTML=''; return; }
    const A = readMatrixA();
    if(hasNaN(A)){ els.warnDD.innerHTML=''; return; }
    const dd = checkDiagonalDominance(A);
    if(!dd.dominant){
      els.warnDD.innerHTML = `<div class="warning-banner">⚠ Matriks belum tentu dominan diagonal — iterasi mungkin tidak konvergen (hasil tetap akan dihitung).</div>`;
    } else {
      els.warnDD.innerHTML = '';
    }
  }

  function fmt(v, d=6){
    if(!isFinite(v)) return '—';
    if(Math.abs(v) < 1e-12) v = 0;
    let s = v.toFixed(d);
    s = s.replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');
    return s;
  }
  function fmtSci(v){
    if(!isFinite(v)) return '—';
    if(v === 0) return '0';
    return v.toExponential(3);
  }

  function renderMatrixStatic(M){
    let html = `<div class="matrix-static"><span class="mbr">[</span><table>`;
    M.forEach(row=>{
      html += '<tr>' + row.map(v=>`<td>${fmt(v,4)}</td>`).join('') + '</tr>';
    });
    html += `</table><span class="mbr">]</span></div>`;
    return html;
  }

  function sparklineSVG(history){
    const w = 380, h = 110, pad = 22;
    if(history.length === 0) return '';
    const errs = history.map(hh=>hh.errAbs);
    const logErrs = errs.map(e => Math.log10(Math.max(e, 1e-15)));
    const minY = Math.min(...logErrs), maxY = Math.max(...logErrs);
    const spanY = (maxY - minY) || 1;
    const n = history.length;
    const realPts = history.map((hh,idx)=>{
      const x = pad + (n===1 ? (w-2*pad)/2 : (idx/(n-1)) * (w-2*pad));
      const y = pad + (1 - (logErrs[idx]-minY)/spanY) * (h-2*pad);
      return [x,y];
    });
    const path = realPts.map((p,i)=> (i===0?'M':'L')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
    const dots = realPts.map((p,i)=> `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${i===realPts.length-1?3.5:2}" fill="${i===realPts.length-1?'var(--success)':'var(--cyan)'}" />`).join('');
    let grid = '';
    for(let gy=0; gy<=2; gy++){
      const yy = pad + (gy/2)*(h-2*pad);
      grid += `<line x1="${pad}" y1="${yy.toFixed(1)}" x2="${w-pad}" y2="${yy.toFixed(1)}" stroke="var(--grid-line)" stroke-width="1" />`;
    }
    return `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" style="overflow:visible;">
      ${grid}
      <path d="${path}" fill="none" stroke="var(--cyan)" stroke-width="1.6" />
      ${dots}
    </svg>`;
  }

  function statusBadge(kind, text){
    return `<span class="status-badge ${kind}"><span class="dot"></span>${text}</span>`;
  }

  function renderInverseResult(A,B){
    const { inv, det, singular } = invertAndDet(A);
    if(singular){
      showError('Matriks A bersifat singular (determinan = 0), sehingga tidak memiliki invers dan sistem tidak dapat diselesaikan dengan metode ini.');
      els.emptyState.style.display='block';
      els.resultsContent.style.display='none';
      return;
    }
    const x = matVec(inv, B);
    const res = residualNorm(A,B,x);

    els.emptyState.style.display='none';
    els.resultsContent.style.display='block';

    const solChips = x.map((v,i)=>`<div class="sol-chip"><div class="k">${VARNAMES[i]||'v'+i}</div><div class="v">${fmt(v,5)}</div></div>`).join('');

    els.resultsContent.innerHTML = `
      ${statusBadge('ok','SOLUSI EKSAK DITEMUKAN')}
      <div class="solution-row">${solChips}</div>
      <div class="metrics-row">
        <div class="metric">Determinan A <b>${fmt(det,6)}</b></div>
        <div class="metric">Galat residual ‖Ax − B‖<sub>∞</sub> <b>${fmtSci(res)}</b></div>
      </div>
      <div class="results-flex">
        <div>
          <div class="subhead">Matriks Balikan A⁻¹</div>
          ${renderMatrixStatic(inv)}
        </div>
        <div>
          <div class="subhead">Verifikasi</div>
          <p class="hint" style="max-width:340px;">
            Solusi dihitung dengan x = A⁻¹B menggunakan eliminasi Gauss-Jordan berpivot parsial (mengurangi galat pembulatan).
            Galat residual di atas mengukur seberapa dekat A·x terhadap B — semakin mendekati nol, semakin akurat solusinya.
          </p>
        </div>
      </div>
    `;
  }

  function renderIterativeResult(A,B,x0,tol,maxIter,method){
    const dd = checkDiagonalDominance(A);
    const runner = method === 'jacobi' ? jacobi : gaussSeidel;
    const out = runner(A,B,x0,tol,maxIter);

    els.emptyState.style.display='none';
    els.resultsContent.style.display='block';

    let badge;
    if(out.diverged) badge = statusBadge('bad','DIVERGEN');
    else if(out.converged) badge = statusBadge('ok', `KONVERGEN — ITERASI KE-${out.iterUsed}`);
    else badge = statusBadge('warn', `BERHENTI (MAKS ${maxIter} ITERASI)`);

    const ddWarn = !dd.dominant
      ? `<div class="warning-banner">⚠ Matriks tidak dominan diagonal secara ketat — konvergensi tidak terjamin oleh teori, meski hasil di atas mungkin tetap mendekati solusi.</div>`
      : '';

    const lastErr = out.history.length ? out.history[out.history.length-1] : null;

    const solChips = out.x.map((v,i)=>`<div class="sol-chip"><div class="k">${VARNAMES[i]||'v'+i}</div><div class="v">${out.diverged ? '—' : fmt(v,5)}</div></div>`).join('');

    let tableRows = '';
    out.history.forEach(hh=>{
      tableRows += `<tr><td>${hh.iter}</td>${hh.x.map(v=>`<td>${fmt(v,5)}</td>`).join('')}<td>${fmtSci(hh.errAbs)}</td></tr>`;
    });
    const tableHead = `<tr><th>Iterasi</th>${out.x.map((_,i)=>`<th>${VARNAMES[i]||'v'+i}</th>`).join('')}<th>Galat |Δx|</th></tr>`;

    els.resultsContent.innerHTML = `
      ${badge}
      ${ddWarn}
      <div class="solution-row">${solChips}</div>
      <div class="metrics-row">
        <div class="metric">Jumlah iterasi <b>${out.iterUsed}</b></div>
        <div class="metric">Galat mutlak akhir <b>${lastErr? fmtSci(lastErr.errAbs) : '—'}</b></div>
        <div class="metric">Galat relatif akhir <b>${lastErr? (lastErr.errRel*100).toFixed(4)+'%' : '—'}</b></div>
        <div class="metric">Toleransi ε <b>${tol}</b></div>
      </div>
      <div class="results-flex">
        <div>
          <div class="subhead">Tabel Proses Iterasi</div>
          <div class="table-scroll">
            <table class="data-table">
              <thead>${tableHead}</thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </div>
        <div>
          <div class="subhead">Kurva Penurunan Galat (skala log)</div>
          <div class="sparkline-wrap">${sparklineSVG(out.history)}</div>
          <p class="sparkline-caption">Setiap titik = galat mutlak maksimum |x_baru − x_lama| pada iterasi tersebut.</p>
        </div>
      </div>
    `;
  }

  /* ===================== VALIDATION + MAIN ACTION ===================== */
  function validateInputs(A,B,x0,tol,maxIter){
    if(hasNaN(A) || hasNaN(B)) return 'Terdapat input matriks/vektor yang bukan angka. Periksa kembali nilai yang dimasukkan.';
    if(state.method !== 'inverse'){
      if(hasNaN(x0)) return 'Nilai tebakan awal (x₀) harus berupa angka.';
      if(isNaN(tol) || tol <= 0) return 'Toleransi error harus berupa angka positif, misal 0.0001.';
      if(!Number.isFinite(maxIter) || maxIter < 1) return 'Maksimal iterasi harus berupa bilangan bulat positif.';
    }
    return null;
  }

  function handleHitung(){
    const A = readMatrixA();
    const B = readVectorB();
    let x0 = [], tol = NaN, maxIter = NaN;
    if(state.method !== 'inverse'){
      x0 = readX0();
      tol = readNumberField(els.tolInput);
      maxIter = parseInt(els.maxIterInput.value, 10);
    }
    const errMsg = validateInputs(A,B,x0,tol,maxIter);
    els.errorArea.innerHTML = '';
    if(errMsg){ showError(errMsg); els.emptyState.style.display='block'; els.resultsContent.style.display='none'; return; }

    if(state.method === 'inverse'){
      renderInverseResult(A,B);
    } else {
      renderIterativeResult(A,B,x0,tol,maxIter,state.method);
    }
  }

  /* ===================== TEMA (GELAP / TERANG) ===================== */
  function applyTheme(theme){
    document.documentElement.setAttribute('data-theme', theme);
    const isDark = theme === 'dark';
    els.themeIcon.textContent = isDark ? '☾' : '☀';
    els.themeLabel.textContent = isDark ? 'Mode Gelap' : 'Mode Terang';
    try{ localStorage.setItem('kalkulator-numerik-theme', theme); }catch(e){ /* abaikan jika penyimpanan tidak tersedia */ }
  }

  function initTheme(){
    let saved = null;
    try{ saved = localStorage.getItem('kalkulator-numerik-theme'); }catch(e){ /* abaikan */ }
    // Default: mode terang (sesuai preferensi dosen), kecuali user pernah memilih gelap.
    applyTheme(saved === 'dark' ? 'dark' : 'light');
  }

  els.themeToggle.addEventListener('click', ()=>{
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  /* ===================== EVENT WIRING ===================== */
  function setMethod(m){
    state.method = m;
    [...els.methodList.querySelectorAll('.method-opt')].forEach(opt=>{
      const active = opt.dataset.method === m;
      opt.classList.toggle('active', active);
      opt.querySelector('input').checked = active;
    });
    els.paramsBox.style.display = (m === 'inverse') ? 'none' : 'flex';
    els.methodHint.textContent = METHOD_HINTS[m];
    clearResults();
    checkDiagonalDominanceLive();
  }

  els.methodList.addEventListener('click', (e)=>{
    const opt = e.target.closest('.method-opt');
    if(!opt) return;
    setMethod(opt.dataset.method);
  });

  els.sizeUp.addEventListener('click', ()=>{
    if(state.n < 5){ state.n++; els.sizeVal.textContent = state.n; buildGrids(); }
  });
  els.sizeDown.addEventListener('click', ()=>{
    if(state.n > 2){ state.n--; els.sizeVal.textContent = state.n; buildGrids(); }
  });

  els.btnHitung.addEventListener('click', handleHitung);

  els.btnReset.addEventListener('click', ()=>{
    buildGrids();
    els.tolInput.value = '0.0001';
    els.maxIterInput.value = '50';
    clearResults();
  });

  els.btnContoh.addEventListener('click', ()=>{
    const ex = EXAMPLES[state.method];
    const n = ex.A.length;
    state.n = n;
    els.sizeVal.textContent = n;
    buildGrids();
    for(let i=0;i<n;i++){
      for(let j=0;j<n;j++){
        els.matrixA.querySelector(`[data-a-i="${i}"][data-a-j="${j}"]`).value = ex.A[i][j];
      }
      els.vectorB.querySelector(`[data-b-i="${i}"]`).value = ex.B[i];
    }
    if(state.method !== 'inverse'){
      for(let i=0;i<n;i++){
        els.x0Row.querySelector(`[data-x0-i="${i}"]`).value = ex.x0[i];
      }
      els.tolInput.value = ex.tol;
      els.maxIterInput.value = ex.maxIter;
    }
    checkDiagonalDominanceLive();
    clearResults();
  });

  // live diagonal-dominance re-check while typing
  document.addEventListener('input', (e)=>{
    if(e.target.matches('[data-a-i]')) checkDiagonalDominanceLive();
  });

  /* ===================== INIT ===================== */
  initTheme();
  buildGrids();
  setMethod('inverse');
})();