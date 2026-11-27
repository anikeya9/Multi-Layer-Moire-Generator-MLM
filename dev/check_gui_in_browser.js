// End-to-end check of the GUI in a real (headless) browser.
//
//   voila notebooks/mlm_gui.ipynb --no-browser --port=8866 &
//   node dev/check_gui_in_browser.js <output_dir>
//
// Clicks through Stack -> Plan -> Run -> Results -> Viewer -> Export, saves one screenshot
// per tab in <output_dir> and prints the page errors (should be []) and the WebGL canvases
// of the 3D viewer. Needs Node.js with the `playwright` package and a Chromium.
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2];
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('http://127.0.0.1:8866/', { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForSelector('text=multi-layer moiré builder', { timeout: 120000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/gui_1_stack.png`, fullPage: true });
  await page.click('text=2 Plan'); await page.click('text=Plan (dry run)');
  await page.waitForSelector('text=s per search', { timeout: 60000 });
  await page.screenshot({ path: `${out}/gui_2_plan.png` });
  await page.click('text=3 Run'); await page.locator('button.widget-button', { hasText: /^\s*Run\s*$/ }).click();
  await page.waitForSelector('text=full-stack candidates passed', { timeout: 300000 });
  await page.screenshot({ path: `${out}/gui_3_run.png` });
  await page.click('text=4 Results'); await page.waitForTimeout(3000);
  await page.waitForSelector('.mlm-candidates select option', { timeout: 60000 }); await page.selectOption('.mlm-candidates select', { index: 0 });
  await page.screenshot({ path: `${out}/gui_4_results.png`, fullPage: true });
  await page.click('text=View selected');
  await page.waitForSelector('text=Lineage', { timeout: 120000 });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: `${out}/gui_5_viewer.png`, fullPage: true });
  const canvases = await page.$$eval('canvas', c => c.map(x => [x.width, x.height]));
  await page.click('text=6 Export'); await page.click('text=Write files');
  await page.waitForSelector('text=stack_spec.yaml', { timeout: 120000 });
  await page.screenshot({ path: `${out}/gui_6_export.png` });
  console.log(JSON.stringify({ errors, canvases }));
  await browser.close();
})().catch(e => { console.error('FAILED', e); process.exit(1); });
