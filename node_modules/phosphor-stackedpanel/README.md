phosphor-stackedpanel
=====================

[![Build Status](https://travis-ci.org/phosphorjs/phosphor-stackedpanel.svg)](https://travis-ci.org/phosphorjs/phosphor-stackedpanel?branch=master)
[![Coverage Status](https://coveralls.io/repos/phosphorjs/phosphor-stackedpanel/badge.svg?branch=master&service=github)](https://coveralls.io/github/phosphorjs/phosphor-stackedpanel?branch=master)

A Phosphor layout panel where visible children are stacked atop one another.

[API Docs](http://phosphorjs.github.io/phosphor-stackedpanel/api/)


Package Install
---------------

**Prerequisites**
- [node](http://nodejs.org/)

```bash
npm install --save phosphor-stackedpanel
```


Source Build
------------

**Prerequisites**
- [git](http://git-scm.com/)
- [node](http://nodejs.org/)

```bash
git clone https://github.com/phosphorjs/phosphor-stackedpanel.git
cd phosphor-stackedpanel
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
  StackedPanel
} from 'phosphor-stackedpanel';

import {
  Widget
} from 'phosphor-widget';


// Create some content for the panel.
let w1 = new Widget();
let w2 = new Widget();
let w3 = new Widget();

// Setup the stacked panel.
let panel = new StackedPanel();
panel.addChild(w1);
panel.addChild(w2);
panel.addChild(w3);

// Toggle the visible widgets as needed.
w1.hide();
w2.show();
w3.hide();
```
