import { execSync } from 'child_process';
import { mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const input = resolve(__dirname, '../src/main.md');
const output = resolve(__dirname, '../out/index.html');
const theme = resolve(__dirname, '../assets/theme.css');

mkdirSync(dirname(output), { recursive: true });

const cmd = `npx marp ${input} --html --theme-set ${theme} --output ${output}`;
console.log(`Running: ${cmd}`);
execSync(cmd, { stdio: 'inherit' });

// Inject Mermaid script
const mermaidInit = resolve(__dirname, '../src/mermaid-init.js');
const mermaidInitContent = readFileSync(mermaidInit, 'utf8');

let html = readFileSync(output, 'utf8');
html = html.replace('</head>', `
  <script src="mermaid.min.js"></script>
  <script>
    ${mermaidInitContent}
  </script>
</head>`);

// Copy Mermaid script to output directory
const mermaidSrc = resolve(__dirname, '../assets/mermaid.min.js');
const mermaidDest = resolve(dirname(output), 'mermaid.min.js');
copyFileSync(mermaidSrc, mermaidDest);

// Copy logos to output directory
const logoDarkSrc = resolve(__dirname, '../assets/fr-logo-horizontal-dark.png');
const logoDarkDest = resolve(dirname(output), 'fr-logo-horizontal-dark.png');
copyFileSync(logoDarkSrc, logoDarkDest);

const logoLightSrc = resolve(__dirname, '../assets/fr-logo-horizontal-light.png');
const logoLightDest = resolve(dirname(output), 'fr-logo-horizontal-light.png');
copyFileSync(logoLightSrc, logoLightDest);

// Write the modified HTML back to the output file
writeFileSync(output, html);

console.log(`✅ Presentation generated at ${output}`);
