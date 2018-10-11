phosphide
=========

[![Build Status](https://travis-ci.org/phosphorjs/phosphide.svg)](https://travis-ci.org/phosphorjs/phosphide?branch=master)
[![Coverage Status](https://coveralls.io/repos/phosphorjs/phosphide/badge.svg?branch=master&service=github)](https://coveralls.io/github/phosphorjs/phosphide?branch=master)

Slightly opinionated scaffolding for building plugin-based IDE-style applications.

[API Docs](http://phosphorjs.github.io/phosphide/api/)


Package Install
---------------

**Prerequisites**
- [node](http://nodejs.org/)

```bash
npm install --save phosphide
```


Source Build
------------

**Prerequisites**
- [git](http://git-scm.com/)
- [node](http://nodejs.org/)

```bash
git clone https://github.com/phosphorjs/phosphide.git
cd phosphide
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

Navigate to `example` and start a server.


Supported Runtimes
------------------

The runtime versions which are currently *known to work* are listed below.
Earlier versions may also work, but come with no guarantees.

- IE 11+
- Firefox 32+
- Chrome 38+
