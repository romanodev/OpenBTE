phosphor-panel
==============

[![Build Status](https://travis-ci.org/phosphorjs/phosphor-panel.svg)](https://travis-ci.org/phosphorjs/phosphor-panel?branch=master)
[![Coverage Status](https://coveralls.io/repos/phosphorjs/phosphor-panel/badge.svg?branch=master&service=github)](https://coveralls.io/github/phosphorjs/phosphor-panel?branch=master)

A convenient Phosphor panel widget and layout.

[API Docs](http://phosphorjs.github.io/phosphor-panel/api/)

Package Install
---------------

**Prerequisites**
- [node](https://nodejs.org/)

```bash
npm install --save phosphor-panel
```

Source Build
------------

**Prerequisites**
- [git](http://git-scm.com/)
- [node](http://nodejs.org/)

```bash
git clone https://github.com/phosphorjs/phosphor-panel.git
cd phosphor-panel
npm install
```

**Rebuild**
```bash
npm run clean
npm run build
```

Run Tests
---------

Follow the source build instructions first.

```bash
npm test
```

Build Docs
----------

Follow the source build instructions first.

```bash
npm run docs
```

Navigate to `docs/index.html`.

Supported Runtimes
------------------

The runtime versions which are currently *known to work* are listed below.
Earlier versions may also work, but come with no guarantees.

- IE 11+
- Firefox 32+
- Chrome 38+

Bundle for the Browser
----------------------

Follow the package install instructions first.

```bash
npm install --save-dev browserify
browserify myapp.js -o mybundle.js
```

Usage Examples
--------------

**Note:** This module is fully compatible with Node/Babel/ES6/ES5. Simply
omit the type declarations when using a language other than TypeScript.

A `Panel` is a convenient `Widget` subclass which acts as a container for
child widgets. Adding child widgets to a panel is simple:

```typescript
import {
  Panel
} from 'phosphor-panel';

import {
  Widget
} from 'phosphor-widget';

let panel = new Panel();
let child1 = new Widget();
let child2 = new Widget();
panel.addChild(child1);
panel.addChild(child2);
```

A more realistic scenario would involve custom widgets and CSS layout:

```typescript
class LogWidget extends Widget {
  ...
}

class ControlsWidget extends Widget {
  ...
}

let logPanel = new Panel();
logPanel.addClass('my-css-layout');

let log = new LogWidget();
log.addClass('log-widget');

let controls = new ControlsWidget();
controls.addClass('controls-widget');

logPanel.addChild(log);
logPanel.addChild(controls);
```

The `Panel` and `PanelLayout` classes make it simple to create container
widgets which cover a vast swath of use cases. Simply add CSS classes to
the panel and child widgets and use regular CSS to control their layout.

Alternatively, these classes may be subclassed to create more specialized
panels and layouts. The PhosphorJS project provides several useful panels
and layouts out of the box. Some of the more commonly used are:

- [BoxPanel](https://github.com/phosphorjs/phosphor-boxpanel)
- [DockPanel](https://github.com/phosphorjs/phosphor-dockpanel)
- [GridPanel](https://github.com/phosphorjs/phosphor-gridpanel)
- [SplitPanel](https://github.com/phosphorjs/phosphor-splitpanel)
- [TabPanel](https://github.com/phosphorjs/phosphor-tabs)
