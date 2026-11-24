---
title: generate_dataset
description: Simulate n_orbits separate CubeSat orbits in a loop, where each orbit generates its own X_orbit and y_orbit with length `(n_bins - background_bins_number - foreground_bins_number + 1)`.
---

<p class="api-meta"><span class="api-badge api-badge-function">function</span> <code>grb_trigger.dataset</code></p>

```python
grb_trigger.dataset.generate_dataset(n_orbits=50, n_bins=1000, probs=[0.4, 0.3, 0.2], background_bins_number=100, foreground_bins_number=10)
```

Simulate n_orbits separate CubeSat orbits in a loop, where each orbit generates its own X_orbit and y_orbit with length `(n_bins - background_bins_number - foreground_bins_number + 1)`.

<p class="api-label">Parameters</p>

<dl class="api-params">
<dt><code>n_orbits</code> <span class="api-type">int, default 50</span></dt>
<dd>

Number of separate orbits to simulate.

</dd>
<dt><code>n_bins</code> <span class="api-type">int, default 1000</span></dt>
<dd>

Number of bins in an orbit, each bin is a photon count recording.

</dd>
<dt><code>probs</code> <span class="api-type">list of floats, default [0.4, 0.3, 0.2]</span></dt>
<dd>

Probabilities for generating a GRB, False alarm or SAA signal.

</dd>
<dt><code>background_bins_number</code> <span class="api-type">int, default 100</span></dt>
<dd>

Number of bins in the background window.

</dd>
<dt><code>foreground_bins_number</code> <span class="api-type">int, default 10</span></dt>
<dd>

Number of bins in the foreground window.

</dd>
</dl>

<p class="api-label">Returns</p>

<dl class="api-params">
<dt><code>X_orbit</code> <span class="api-type">numpy.ndarray (n, 7)</span></dt>
<dd>

The feature matrix compiled with the results of the
simulation module

</dd>
<dt><code>y_orbit</code> <span class="api-type">numpy.ndarray (n, )</span></dt>
<dd>

The target array with the information returned by generate_labels

</dd>
</dl>

<p class="api-label">Raises</p>

<dl class="api-params">
<dt> <span class="api-type">ValueError</span></dt>
<dd>

If any value in `probs` is larger than 1.0.

</dd>
<dt> <span class="api-type">ValueError</span></dt>
<dd>

If n_bins-`background_bins_number`-`foreground_bins_number` is
less than 1.

</dd>
<dt> <span class="api-type">ValueError</span></dt>
<dd>

If `n_orbits` is smaller than 1.

</dd>
</dl>

<p class="api-label">Notes</p>

Each orbit has a signal with length n_bins.

The length each orbit signal after applying the background
and foreground cuts (length of the output) `n = n_bins -
background_bins_number - foreground_bins_number + 1`.

The probabilities `probs` are not physically correction, in reality
we should detect around 1 GRB signal per 1000 orbits. However, for
the initial purposes of trainign the ML model, we want to boost our
dataset.
