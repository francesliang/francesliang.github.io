#!/usr/bin/env bash
set -e # halt script on error


# build site with jekyll
jekyll build

# cleanup
rm -rf master

# copy generated site to the blog branch
cp  -R _site/* master

#clone `master' branch of the repository using encrypted GH_TOKEN for authentification
git clone https://${GH_TOKEN}@github.com/francesliang/francesliang.github.io.git master

# commit and push changes
cd master
git config user.email "francisliang2010@hotmail.com"
git config user.name "Xin(Frances) Liang"
git add -A .
git commit -am "Travis #$TRAVIS_BUILD_NUMBER"
git push --quiet origin master > /dev/null 2>&1
