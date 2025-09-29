#!/bin/bash

for f in Conllus/*; do cat ${f} | perl -pe 's/\s\s+//g' > tmp; mv tmp ${f}; done