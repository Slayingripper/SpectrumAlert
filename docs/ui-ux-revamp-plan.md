# SpectrumAlert UI/UX Revamp — Plan

Status: **complete** (foundation, shell, theming, and the final audit pass;
see [Rollout status](#10-rollout-status)).

## 1. Goal

Turn the single-page dashboard from a stylised "demo" into a professional,
dense operator console:

- The things that matter (system health, live spectrum, threats, alerts) read
  first, in a predictable grid.
- One top toolbar carries navigation, live status, and global tools.
- A neutral, high-contrast palette that works in a control room and in
  daylight, with no decorative gradients or glows.
- Nothing is faked: labels stay honest (dBFS, "no live link"), and
  every control that is wired behaves as advertised.

Non-goals: changing backend behaviour, renaming APIs, or altering the analysis
logic. This is a presentation-layer refactor that must keep every existing
contract (element ids, asset order, test suite) intact.

## 2. Guiding principles

1. **Neutral, not neon.** One accent (blue) plus four severity colours. Flat
   surfaces, 1px borders, subtle elevation. No drop-shadow glows, no radial
   background gradients, no drop-shadow pulses.
2. **Information hierarchy over decoration.** Data leads; chrome recedes.
   Panel headers are quiet, numbers and status are emphasised.
3. **Consistency through tokens.** Colour, spacing, radius, and type come from
   one token set. Components reference tokens, never raw hex.
4. **Honesty is part of the design.** dBFS units, stale/error
   states, and "not connected" copy are treated as first-class UI.
5. **Accessible by construction.** Visible focus, reduced-motion support, no
   colour-only encoding, labelled dialogs and canvases.
6. **Contract-safe.** All element ids referenced by the JS remain, asset load
   order is preserved, and the static UI contract tests stay green.

## 3. Problems in the current UI

- **Corrupted template.** A duplicated `</body>`/scripts/inline block and
  literal `%s` placeholders were appended after the first document; a second
  copy of the inline tab script had a brace mismatch (a real
  `missing ) after argument list` parse error at load).
- **Invisible icons everywhere.** `<i class="fas fa-…">` is used ~35 times but
  Font Awesome is never loaded, so panel headers and buttons had blank icon
  slots.
- **Two competing token systems.** Shell tokens (`--bg-*`, `--accent-*`) live in
  `cyberpunk.css`; the workspace uses a separate `--sa-*` set in
  `geolocate.css`. Light mode only overrode half of them, so `--accent` stayed
  neon in light mode.
- **Broken/duplicated modals.** `showFileAnalysis` and `displayDatasetSelector`
  injected a second DOM node with the *same id* as the static modal and
  hard-coded neon inline styles; closing one removed the wrong node.
- **Unstyled components.** `.analytics-grid`, `.analytics-metric`,
  `.modal-overlay`, `.dataset-item`, `.file-details`, etc. had no CSS at all.
- **Layout waste.** The dashboard tab stacked every panel full-width; the
  sidebar ate 200px; a scroll container's overflow leaked ~1700px of empty
  document height below the footer.
- **"AI-generated" look.** Skull logo, pink/cyan/purple glows, Orbitron all
  caps, gradient panels, blinking LEDs.

## 4. Design system

### 4.1 Palette (dark default)

| Token | Value | Use |
| --- | --- | --- |
| `--bg-primary` | `#0b0e14` | app background |
| `--bg-secondary` | `#141922` | panels, app bar, footer |
| `--bg-tertiary` | `#1b212b` | insets, inputs, chips |
| `--bg-raised` | `#1f2632` | hover surfaces |
| `--border-color` | `#262d38` | 1px dividers |
| `--border-strong` | `#333c4a` | hover borders |
| `--text-primary` | `#e6ebf2` | body / values |
| `--text-secondary` | `#aab4c0` | secondary copy |
| `--text-muted` | `#8a94a6` | labels (≥4.5:1 on both surfaces) |
| `--accent` | `#4c8dff` | primary accent |
| `--success` | `#46b450` | ok / running |
| `--warning` | `#d9a441` | pending / caution |
| `--danger` | `#ef5350` | critical / stop |

Filled accent/severity surfaces get text from dedicated `--on-*` tokens
(near-black on dark surfaces, white on light) so filled buttons and chips keep
their contrast in both themes.

### 4.2 Palette (light)

`#f4f6fa` / `#ffffff` / `#eceff4` surfaces, `#141922` ink, `#1553c2` accent,
`#146c2e` / `#7a5200` / `#cf222e` status. The light set is defined once in
`ui-core.css` under `html[data-theme="light"]`.

### 4.3 Type

- UI: system sans stack (`-apple-system, "Segoe UI", Roboto, …`). Orbitron and
  Rajdhani are dropped (and the Google Fonts request removed).
- Numerals / code: system mono stack (`--sa-font-mono`).
- Scale: 14px body, 0.68–0.8rem labels (uppercase, letter-spaced, muted),
  1–1.7rem for metric values.

### 4.4 Shape and motion

- Radii 6–10px; 1px borders; elevation via one soft shadow token.
- Motion only for state changes (≤240ms); disabled under
  `prefers-reduced-motion`.

## 5. Layout & information architecture

### 5.1 Top toolbar

Replace the header + 200px sidebar with a single sticky application bar:

- **Left** — brand mark, wordmark (`Spectrum` + accent `Alert`), "RF monitoring"
  tag.
- **Centre** — section navigation as `.nav-tab` buttons: Dashboard, Config,
  Training, Workspace (same ids/classes the tab script and audit use).
- **Right** — live status pills (System / Monitor / Threats) then global tools
  (jump-to-workspace, theme, mute, shortcuts).

### 5.2 Dashboard grid

Each tab is a 12-column CSS grid; panels declare a span so the important,
wide content leads:

| Panel | Span |
| --- | --- |
| Getting started (`sa-guide`) | 12 |
| Spectrum analysis | 8 |
| System status | 4 |
| Threat detection | 7 |
| Recent alerts | 5 |
| Real-time analytics | 12 |
| Config / Data, Model / Training | 7 / 5 |
| Workspace section | 12 |

Panels stretch to equal height (`align-items: stretch` + `height: 100%`), and
everything collapses to a single column below 1100px.

### 5.3 Modes and states

Loading, empty, error, and stale states already exist in `ui-core.css`
(`.sa-state--empty|error|stale`); the revamp keeps and restyles them so a quiet
band never looks like a broken one.

## 6. Component work

- **App bar**: new `.app-bar`, `.app-bar__brand/nav/status/tools`,
  `.status-pill`, updated `.led` (active/inactive).
- **Panels**: flat surface, quiet header, consistent padding, equal heights,
  collapsible toggle restyled.
- **Buttons/inputs**: one base style for the many ad-hoc button classes
  (`btn-refresh`, `btn-start`, `btn-train`, …), token-driven, with `--on-*`
  text; unified `control-select` / config inputs.
- **Metrics/stats**: card grid with mono values; system and analytics
  metric cards unified.
- **Charts**: Chart.js colours read from CSS tokens via a `chartColors` helper
  (no baked neon); rebuilt on theme change.
- **Icons**: all invisible Font Awesome slots removed; icon-only controls use a
  glyph plus `aria-label` (e.g. refresh `⟳`, clear `✕`, theme `◐`).
- **Data file list**: real layout with separated metadata
  (size · samples · band · date); dead `.file-icon` slot dropped.
- **Modals**: one static, labelled dialog per job. `showFileAnalysis` /
  `displayDatasetSelector` now populate the existing `#file-analysis-body` /
  `#dataset-list` and toggle the overlay instead of injecting duplicate ids and
  neon markup.
- **Workspace**: adopts the shared palette through the `--sa-*` tokens; primary
  buttons use `--on-accent`; PMR cells use translucent tints that survive light
  mode.

## 7. Accessibility and honesty constraints

- Skip link, labelled dialogs, `role="img"` + `aria-label` on canvases.
- `:focus-visible` everywhere; `prefers-reduced-motion` respected.
- Severity is never colour-only (glyphs + text badges).
- Status copy stays literal: "Saved (no live link)", "No SDR",
  dBFS.
- Every text run verified ≥4.5:1 (≥3:1 for large) in both themes.

## 8. Technical guardrails

- **Element ids** referenced by `dashboard.js` must exist in `dashboard.html`
  (`test_no_missing_element_ids`) and must be unique
  (`test_no_duplicate_ids`).
- **Asset order**: bootstrap → cyberpunk → ui-core → geolocate CSS;
  bootstrap.bundle → ui-core.js → geolocate.js → dashboard.js →
  dashboard-extras.js JS. Bootstrap stays vendored; leaflet + heat + geolocate
  assets stay.
- Full-viewport layout (`max-width: none`), map scales with the viewport.
- No dead scripts loaded; no `Power (dBm)` labels.

## 9. Verification

1. `python3 -m pytest -q` — full suite (target: no regressions).
2. `node --check` on every changed JS file.
3. WCAG contrast sweep over every rendered text run, dark and light
   (target: 0 failing runs).
4. Playwright audit: tab switching, receivers, monitoring shortcut, alerts,
   overflow at 390/768/1100/1920/2560px, console errors.
5. Browser screenshots across tabs, both themes, and mobile width.

## 10. Rollout status

- [x] Repair the corrupted `dashboard.html` (dedupe, remove `%s`, fix the
      inline-script brace error).
- [x] New top toolbar; sidebar removed; `.nav-tab` preserved.
- [x] Shared neutral design system in `cyberpunk.css`; light tokens in
      `ui-core.css`; workspace tokens in `geolocate.css`.
- [x] Dashboard 12-column grid with sensible panel spans and equal heights.
- [x] Drop Orbitron/Rajdhani + Google Fonts; system font stacks.
- [x] Remove all invisible Font Awesome icons; glyph + label for icon-only
      controls.
- [x] Token-driven Chart.js colours and `dBFS` labels.
- [x] Refactor the two dynamic modals onto the static, themeable markup.
- [x] Fix the scroll-container overflow that stretched the page.
- [x] Contrast: 0 failing runs in both themes; full test suite green.
- [x] Final pass: run the full Playwright audit, re-review screenshots, and do
      a last manual browser check at 390–2560px.

### 10.1 Final pass results

**Audit.** `python3 /tmp/opencode/audit.py` (recreated after `/tmp` was wiped;
starts its own dev server, or takes `--base-url`). Report:
`/tmp/opencode/audit-report.json`, screenshots: `/tmp/opencode/shots/`.
Result: **40/40 checks** — tab switching, receiver add/cleanup, the `S`
monitoring shortcut (start and stop), alerts (empty state, severity filter,
ack-all, export, clear), theme toggle, a11y spot checks, zero console/page
errors, and horizontal overflow across 4 tabs × 390/768/1100/1920/2560px.

**Issues found and fixed** (CSS plus four canvas labels; no element ids, no
test contracts, no JS changes):

- `#detections-timeline-chart`, `#frequency-distribution-chart`,
  `#model-performance-chart`, `#sa-waterfall` had no `role="img"` /
  `aria-label`, so three of the dashboard's charts and the waterfall were
  silent to a screen reader.
- The sticky app bar wrapped to four rows at 390px and ate 174px of the 844px
  viewport. Ordering the tools beside the brand below 505px keeps it to three
  rows (140px); 505px and up is unchanged.
- `.config-item input[type=range]` was a 6px-tall hit area. The track still
  renders at 6px (centred in the input, verified pixel-for-pixel in both
  themes); the input box is now 24px with a transparent background.
- Target sizes: `.sa-alert__action` 22px → 24px, `.sa-btn.tiny` ≥24px, and both
  added to the existing `≤768px` 40px touch-target rule (`.sa-btn.tiny` gets
  its own media block in `geolocate.css`, which loads after `ui-core.css`).

**Deliberately left alone:** 10–10.9px labels (the §4.3 0.68–0.8rem band),
native checkbox sizes, and standalone hint links (WCAG 2.5.8 spacing
exception). `POST /api/spectrum` returning 503 with no SDR attached is the
honest backend answer, so the audit records it as expected rather than a UI
defect.

**Verification:** `python3 -m pytest -q` → 392 passed, 3 skipped;
`node --check` clean on every static JS file; no colour tokens were touched,
so the contrast sweep is unchanged. Screenshots were regenerated for all tabs,
both themes, 390/1920/2560px and reviewed programmatically (non-flat, theme
luminance delta >179) — this session's model cannot view images.

### 10.2 Workspace tab recovery, layout fixes and guided mode

After §10.1 a data-loss incident reverted `dashboard.html` to the committed
pre-revamp file (the app-bar/tabs build was uncommitted). The app-bar + four-tab
layout was rebuilt from the intermediate `dashboard.html.backup` and now lives
in `dashboard.html`; `.backup` is kept for reference. All other changes
(CSS/JS) were untouched by the incident.

**Workspace layout fixes** (the "squished" workspace):

- `.sa-layout { align-items: stretch }` (was `flex-start`) so the waterfall and
  the side panels share the row height.
- Sticky pipeline rail: `top: 72px`, `max-height: calc(100vh - 96px)` at
  ≥1400px.
- `#sa-map { height: clamp(360px, 56vh, 820px); min-height: 320px }` instead of
  a fixed height that squeezed the emission map.
- Button consistency: `.sa-btn` now matches the dashboard `.btn-*` base
  (7px 12px padding, 7px radius, 0.8rem/600, 32px min-height, same
  hover/border tokens). `.sa-input`/`.sa-select` keep their compact sizing.

**Overflow regression fixed:** `.sa-readout` inherited `white-space: nowrap`
from the design system's crosshair-chip rule (`ui-core.css`), so the
workspace's stat readout under the waterfall could not wrap and a long state
string blew the 390px viewport. `geolocate.css` now restores `white-space:
normal` for the stat readouts and scopes `nowrap` back to the absolute cursor
chips (`.sa-chart-shell .sa-readout`, `#sa-cursor-readout`).

**Guided mode** (requested: one-at-a-time walkthrough for triangulate /
fingerprint / anomalies):

- The getting-started strip is now six steps: add receivers → monitor →
  triangulate → identify → review results → review anomalies. The three
  original ids (`sa-step-analyse`, `sa-step-analyse-status`,
  `sa-guide-workspace`) are preserved as the "Review the results" step, so the
  test contract is unchanged.
- "Guided mode" toggle in the guide head (`#sa-guide-mode`): `data-mode="tour"`
  on the guide hides every step except the current one, and a walk-through row
  (`#sa-guide-tour`) with a progress bar, counter ("Step 2 of 6"), Back/Next
  (`#sa-guide-prev`/`#sa-guide-next`) appears. The choice persists in
  `localStorage` (`SAUI.store`).
- Every step action now activates the Workspace tab before scrolling
  (`window.activateTab` is exposed by the inline tab script; the guide's
  workspace buttons are in the delegated handler), which fixes latent
  hidden-scroll-target bugs from before the tab redesign.
- Guided focus is now the default view: unless the user has previously exited
  it (the choice is persisted), `body.sa-tour` hides every `.panel` / `.sa-panel`
  on the page so the app opens as the guide alone, and each step opens only the
  panels it needs (receivers on the first, spectrum on "monitor", waterfall +
  emitter map on "triangulate", waterfall + identify on "fingerprint", analytics
  on "review", anomaly + alerts on "anomalies"). Hiding the guide mid-tour exits
  the tour so the page never goes blank, and a tour stored from a previous
  session is not restored while the guide is hidden. Guide buttons use the
  standard full-size `.sa-btn` style so they match the dashboard controls.
- `refreshGuide()` writes states for all six steps from live state (roster
  count, monitoring flag, `#sa-estimate`/`#sa-fp-verdict` results,
  `#alerts-list`).

**Verification after guided mode:** full `pytest` → 392 passed, 3 skipped;
`node --check` clean on all static JS; Playwright audit → 40/40 (all widths ×
all tabs, console/page error free); a 31-check guided-mode probe and a
28-check panel-focus probe exercised the tour (toggle on/off, walking all six
steps, per-step action jumps and panel reveal/hide on both the dashboard and
the workspace, exit/restore, hide-guide-exits-tour) — all passing. No colour
tokens changed, so the contrast sweep remains at the §10.1 result.
