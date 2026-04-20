#!/usr/bin/env bash

set -euo pipefail

with test test/core_test.w
with test test/view_test.w
with test test/ir_test.w
with test test/ir_text_test.w
with test test/cpu_interp_test.w
with test test/kernel_source_test.w
with test test/kernel_cpu_test.w
with test test/runtime_stub_test.w
with test test/view_smoke.w
