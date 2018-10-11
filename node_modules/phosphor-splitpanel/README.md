phosphor-splitpanel
===================

[![Build Status](https://travis-ci.org/phosphorjs/phosphor-splitpanel.svg)](https://travis-ci.org/phosphorjs/phosphor-splitpanel?branch=master)
[![Coverage Status](https://coveralls.io/repos/phosphorjs/phosphor-splitpanel/badge.svg?branch=master&service=github)](https://coveralls.io/github/phosphorjs/phosphor-splitpanel?branch=master)

A Phosphor layout panel which arranges its children into resizable sections.

[API Docs](http://phosphorjs.github.io/phosphor-splitpanel/api/)


Package Install
---------------

**Prerequisites**
- [node](http://nodejs.org/)

```bash
npm install --save phosphor-splitpanel
```


Source Build
------------

**Prerequisites**
- [git](http://git-scm.com/)
- [node](http://nodejs.org/)

```bash
git clone https://github.com/phosphorjs/phosphor-splitpanel.git
cd phosphor-splitpanel
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
# run tests in Firefox
npm test

# run tests in Chrome
npm run test:chrome

# run tests in IE
npm run test:ie
```


Build Docs
----------

Follow the source build instructions first.

```bash
npm run docs
```

Navigate to `docs/index.html`.


Build Example
-------------

Follow the source build instructions first.

```bash
npm run build:example
```

Navigate to `example/index.html`.


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
npm install --save-dev browserify browserify-css
browserify myapp.js -o mybundle.js
```


Usage Examples
--------------

**Note:** This module is fully compatible with Node/Babel/ES6/ES5. Simply
omit the type declarations when using a language other than TypeScript.

```typescript
import {
  SplitPanel
} from 'phosphor-splitpanel';

import {
  Widget
} from 'phosphor-widget';


// Create some content for the panel.
let w1 = new Widget();
let w2 = new Widget();
let w3 = new Widget();

// Set the widget stretch factors (optional).
SplitPanel.setStretch(w1, 0);
SplitPanel.setStretch(w2, 2);
SplitPanel.setStretch(w3, 1);

// Setup the split panel.
let panel = new SplitPanel();
panel.handleSize = 5;
panel.orientation = SplitPanel.Horizontal;
panel.addChild(w1);
panel.addChild(w2);
panel.addChild(w3);

// sometime later...

// Get the normalized relative widget sizes.
let sizes = panel.sizes();

// Set the relative widget sizes.
panel.setSizes([2, 4, 1]);
```
