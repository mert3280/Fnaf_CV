/* Capture the eval-dashboard UI screenshots for the Week-5 deliverable.
 *
 * Drives the REAL dashboard (started separately with --no-tracker, so the
 * scripted mouse stands in for the hand controller) through every screen and
 * writes PNGs to DOCS/class-related/week5/ui-screenshots/.
 *
 *   python -m src.eval_dashboard.server --no-tracker --no-browser --port 5057 \
 *          --results-dir <scratch>            # in another terminal
 *   node scripts/capture_ui_screenshots.js [--port 5057] [--out <dir>]
 *
 * Needs the globally installed playwright package:
 *   $env:NODE_PATH = "$(npm root -g)"
 *
 * Screens 05-10 are captured in --no-tracker mode: the obstacle logic, timing
 * and scoring are the real ones, but the pointer is Playwright's synthetic
 * mouse, not the hand tracker. Labelled as such in ui-walkthrough.md.
 */
"use strict";

const path = require("path");
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};

const PORT = argOf("--port", "5057");
const BASE = `http://127.0.0.1:${PORT}/`;
const OUT = path.resolve(argOf("--out", "DOCS/class-related/week5/ui-screenshots"));
const VIEW = { width: 1600, height: 950 };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function shot(page, name) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file });
  console.log(`  wrote ${name}`);
}

/** Move the mouse to the centre of an element and click it. */
async function clickCenter(page, selector) {
  const box = await page.locator(selector).first().boundingBox();
  if (!box) throw new Error(`no box for ${selector}`);
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 12 });
  await page.mouse.down();
  await sleep(60);
  await page.mouse.up();
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: VIEW, deviceScaleFactor: 1 });
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`  [console error] ${m.text()}`);
  });

  console.log(`capturing ${BASE} -> ${OUT}`);
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector("#startBtn:not([disabled])");   // --no-tracker opens it
  await page.mouse.move(VIEW.width * 0.5, VIEW.height * 0.72, { steps: 5 });
  await sleep(300);

  /* ---- welcome tabs ---- */
  await shot(page, "01-start-tab.png");

  await clickCenter(page, "#tabOverview");
  await page.waitForSelector("#ovTable tbody tr");
  await sleep(500);
  await shot(page, "02-overview-model-metrics.png");

  // scroll the fixed .screen container down to the comparison chart
  await page.evaluate(() => {
    const el = document.getElementById("screen-welcome");
    el.scrollTop = document.getElementById("ovCharts").offsetTop - 80;
  });
  await sleep(400);
  await shot(page, "03-overview-comparison-chart.png");

  // Same tab, filtered + re-charted + re-sorted, so the screenshot shows the
  // controls doing something rather than repeating the view above.
  await page.selectOption("#ovModel", { label: "fistvsrest_frozen_mnv3_large" });
  await page.selectOption("#ovBarMetric", "ob3Accuracy");
  await page.selectOption("#ovColorBy", "strategy");
  await page.selectOption("#ovSort", "ob3Accuracy");
  await page.selectOption("#ovSortDir", "worst");
  await sleep(500);
  await page.evaluate(() => {
    const el = document.getElementById("screen-welcome");
    el.scrollTop = document.getElementById("ovCharts").offsetTop - 80;
  });
  await sleep(400);
  await shot(page, "04-overview-filtered-and-sorted.png");

  await clickCenter(page, "#ovClear");
  await page.selectOption("#ovSort", "date");
  await page.selectOption("#ovSortDir", "best");
  await page.evaluate(() => { document.getElementById("screen-welcome").scrollTop = 0; });
  await clickCenter(page, "#tabHelp");
  await sleep(400);
  await shot(page, "05-help-tab.png");

  /* ---- run the course ---- */
  await clickCenter(page, "#tabStart");
  await sleep(200);
  await clickCenter(page, "#startBtn");
  await page.waitForSelector("#screen-ob1.active .num-btn");

  // clear two targets so the progress counter shows a run in flight
  for (let i = 0; i < 2; i++) {
    await clickCenter(page, "#ob1Box .num-btn");
    await sleep(250);
  }
  await page.waitForSelector("#ob1Box .num-btn");
  const b3 = await page.locator("#ob1Box .num-btn").first().boundingBox();
  await page.mouse.move(b3.x - 140, b3.y + 90, { steps: 16 });   // pointer visibly en route
  await sleep(300);
  await shot(page, "06-obstacle1-precision.png");

  while (await page.locator("#ob1Box .num-btn").count()) {
    await clickCenter(page, "#ob1Box .num-btn");
    await sleep(200);
  }

  /* ---- obstacle 2: rapid clicks ---- */
  await page.waitForSelector("#screen-ob2.active");
  await sleep(200);
  for (let i = 0; i < 4; i++) { await clickCenter(page, "#downBtn"); await sleep(140); }
  await sleep(200);
  await shot(page, "07-obstacle2-rapid-clicks.png");
  for (let i = 0; i < 3; i++) { await clickCenter(page, "#downBtn"); await sleep(140); }

  /* ---- obstacle 3: tracer ---- */
  await page.waitForSelector("#screen-ob3.active");
  await sleep(500);
  await shot(page, "08-obstacle3-tracer-preview.png");

  await clickCenter(page, "#traceStartBtn");
  await sleep(200);
  const geom = await page.evaluate(() => ({
    pts: ob3.pts.map((p) => ({ x: p.x, y: p.y })),
    left: ob3.svgRect.left,
    top: ob3.svgRect.top,
  }));
  // Trace the path with a small deliberate wobble so the accuracy score is a
  // real measurement of a real (if synthetic) trace, not a perfect 100.
  let shotTaken = false;
  for (let i = 0; i < geom.pts.length; i++) {
    const p = geom.pts[i];
    const wob = 9 * Math.sin(i * 1.7);
    await page.mouse.move(geom.left + p.x, geom.top + p.y + wob, { steps: 4 });
    if (!shotTaken && i > geom.pts.length * 0.55) {
      await sleep(150);
      await shot(page, "09-obstacle3-tracer-trail.png");
      shotTaken = true;
    }
  }

  /* ---- finish gate, survey, done ---- */
  await page.waitForSelector("#screen-finish.active", { timeout: 5000 });
  await sleep(400);
  await shot(page, "10-finish-summary.png");

  await clickCenter(page, "#finishBtn");
  await page.waitForSelector("#screen-survey.active .q");
  // answer all five 1-5 questions (pick a spread, not all 5s)
  const picks = [4, 4, 5, 5, 4];
  for (let q = 0; q < picks.length; q++) {
    await clickCenter(page, `.q:nth-of-type(${q + 1}) .scale button:nth-child(${picks[q]})`);
    await sleep(120);
  }
  await sleep(300);
  await shot(page, "11-survey.png");

  await clickCenter(page, "#submitBtn");
  await page.waitForSelector("#restartBtn:not([hidden])", { timeout: 8000 });
  await sleep(400);
  await shot(page, "12-saved.png");

  const msg = await page.locator("#doneMsg").textContent();
  console.log(`  done screen says: ${msg}`);
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
