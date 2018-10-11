jupyter-js-utils
================

[![Build Status](https://travis-ci.org/jupyter/jupyter-js-utils.svg)](https://travis-ci.org/jupyter/jupyter-js-utils?branch=master)
[![Coverage Status](https://coveralls.io/repos/jupyter/jupyter-js-utils/badge.svg?branch=master&service=github)](https://coveralls.io/github/jupyter/jupyter-js-utils?branch=master)

JavaScript utilities for the Jupyter frontend.

[API Docs](http://jupyter.github.io/jupyter-js-utils/)


Package Install
---------------

**Prerequisites**
- [node](http://nodejs.org/)

```bash
npm install --save jupyter-js-utils
```


Source Build
------------

**Prerequisites**
- [git](http://git-scm.com/)
- [node](http://nodejs.org/)

```bash
git clone https://github.com/jupyter/jupyter-js-utils.git
cd jupyter-js-utils
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


Supported Runtimes
------------------

The runtime versions which are currently *known to work* are listed below.
Earlier versions may also work, but come with no guarantees.

- Node 0.12.7+
- IE 11+
- Firefox 32+
- Chrome 38+

Note: "requirejs" must be included in a global context for Comm targets.


Usage Examples
--------------

**Note:** This module is fully compatible with Node/Babel/ES6/ES5. Simply
omit the type declarations when using a language other than TypeScript.
