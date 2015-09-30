#!/usr/bin/env bash
set -e # halt script on error


# build site with jekyll
jekyll build

# copy generated site to the blog branch
cp  -f _site/* ../francesliang.github.io

# commit and push changes
cd ../francesliang.github.io
git add -A .
git commit -am "Travis #$TRAVIS_BUILD_NUMBER"
git push
