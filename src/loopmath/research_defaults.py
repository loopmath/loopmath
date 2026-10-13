"""Defaults of the E1 research verbs, kept free of heavy imports: the CLI reads them for every
command's parser, and importing the research fit would pull in pandas."""

SEED = 20260901
E1A_OBSERVE = "luna:low,medium,xhigh;sol:medium;terra:medium"
