---
layout: post_entry
title: The Development of This Blog (II)
---

Following the previous article on "The development of this blog (I)", the second part of the article will discuss the following components:

* [Static Site Generator - Jekyll](#static-site-generator)

* [Continuous Integration - Travis CI](#continuous-integration)


<br>

#### <a name="static-site-generator"></a>Static Site Generator - Jekyll

<br>

At this point, I have a website that includes HTML files for the structure and content, and CSS files for the style and layout. However, each time I want to update an existing post or add a new one, I will have to edit the HTML files or create a new one and copy the structure as well as the layout across respectively, which is not efficient at all. Ideally, I would like to separate the content from the structure and style of the web page, so I will only need to focus on the content when I try to update the blog.

This is why static site generator is useful. It is a software that takes text and templates as input and generates HTML files as output to be displayed on the web browsers. The text is the content that we would like to write in our blog, while the templates are the structure, layout and style etc. that don't require much changes over the updates of the web pages. 

[*Here*](https://www.staticgen.com/) shows the top open-source static site generator and status of their popularity among users. I chose Jekyll to generate the static web pages for my blog because it is on the top of the list and has a very large user base.

**Jekyll**

Jekyll is [*a simple blog-aware static generator*](http://jekyllrb.com/docs/home/) written in Ruby. Once Jekyll is installed, it can build the website project and output files into a specific destination folder (*"_site"* by default). Jekyll also comes with a built-in development server, where we can preview what the generated site looks like locally.

A basic Jekyll site usually has the following folders and files in the project folder:

+ _config.yml - the file that stores configuration settings, including where things are, handling reading, filtering content, plugins, conversion, serving, outputting, and markdown processors etc.

+ _includes - the folder that contains partials that can be reused by layout templates and posts, for example, the footer or the Disqus commenting system.

+ _layouts - the folder that includes the layout templates of posts. The templates are selected by posts with a [*YAML Front Matter*](http://jekyllrb.com/docs/frontmatter/) at the start of a file.

+ _posts - this is the folder where the content of the website written a selected markup language (I used *Markdown* in this blog) sits. The file-names must follow this format: **YEAR-MONTH-DAY-title.MARKUP** while words are separated by hyphens. The [*permalinks*](http://jekyllrb.com/docs/permalinks/) can be customised in the file ```_config.yml``` or each post, but the date and markup language are determined by the file name. At the start of each post file, a YAML Front Matter should be included to indicate which layout template this post is wrapped by. Here is an example of the YAML Front Matter:

		---  
		layout: post_entry  
		title: The Development of This Blog (II)  
		---  

+ index.html - the default file of the web page. With a YAML Front Matter, it can be transformed by Jekyll. Indices of blog posts can be displayed in this file using [*Liquid template language*](https://docs.shopify.com/themes/liquid-documentation/basics). 

+ _site - this folder is where the generated site will be after ```jekyll build``` putting all the layouts and content together. The files in this folder are those needed to display the website in a browser.

Once the files and folders are in place, we could execute Jekyll to generate the site and deploy the files in folder _site to GitHub or other web hosting service:

1. Generate files for the website in the project folder:

		~$ jekyll build


2. Start a development server locally to see what the site will look like:

		~$ jekyll serve

3. Deploy files in folder *_site* to *username.github.io* once the updates are finished. 

**Markdown**

According to the [*Markdown project page*](https://daringfireball.net/projects/markdown/):

>"Markdown is a text-to-HTML conversion tool for web writers. Markdown allows you to write using an easy-to-read, easy-to-write plain text format, then convert it to structurally valid XHTML (or HTML)."

>"Thus, “Markdown” is two things: (1) a plain text formatting syntax; and (2) a software tool, written in Perl, that converts the plain text formatting to HTML."

Users are able to read the text-content easily through the plain text formatting [*syntax*](https://daringfireball.net/projects/markdown/syntax), rather than trying to find the actual content between a group of HTML tags. Then the software tool can convert the plain text format to structurally valid HTML for displaying the web page.

If we use the text editor [*Sublime*](http://www.sublimetext.com/) to edit a Markdown (.md) file, there is also [*Monokai Extended*](https://github.com/jonschlinkert/sublime-monokai-extended), which has additional syntax highlighting for easier editing.

<br>

#### <a name="continous-integration"></a>Continuous Integration - Travis CI

<br>

Up to this point, I have my blog hosted on GitHub and whenever I want to add a new post or edit an existed one, I just need to modify the content files in the *_post* folder and build it without worrying about the layouts and structures of the post pages. However, the files of the site are generated in *_site* folder by default using Jekyll, while GitHub hosts the site based on the files in *username.github.io*. Therefore, I have to build the Jekyll files and copy them to the root directory every time I make a change to the website in the project folder, which can be further automated by Continuous Integration (CI).

Continuous Integration is actually a practice for software development to integrate working copies of all developers' codes into a shared repository several times a day. It uses a revision control system to track the project's source code. Each integration is validated by an automated build, including self-testing to detect any errors or confirm successful integrations. [*Here*](http://martinfowler.com/articles/continuousIntegration.html) is a very detailed explanation of the concept.  

For my blog, my requirements of the tool for continuous integration are simple:

+ Open source and free

+ Allow integration with GitHub since that's where my blog is hosted

+ It's hosted so I have flexible access via Internet

**Travis CI**

I chose Travis CI as my continuous integration tool because it has what I need for this blog. 

> "Travis CI is a free and open-source software, hosted, distributed continuous integration service used to build and test projects hosted at GitHub.""

Once we sign in to Travis CI with our GitHub account, we could enable Travis CI for the project repository that we want to build. Then add a configuration file - ```.travis.yml``` to our repository as an instruction on what and how to build for Travis CI. If the setup is correct, Travis CI will trigger a build every time when we commit and push a change to the corresponding repository.

The basic steps of the integration, after we've enabled Travis CI in our profile page for our repository, are as following:

1. GitHub Branch Set-up

	I have two branches for my blog repository on GitHub:

	* ```master``` - for the content of the generated static site (the content in the *_site* folder)

	* ```website-template``` - for the Jekyll configuration and markup sources

	The purpose is for Travis CI to monitor the ```website-template``` branch. Whenever  a change is committed to the branch, ```jekyll build``` can be run to process the markup sources and generate the site content. Travis CI can then copy the content to the ```master``` branch.


2. Travis CI Configuration

	The configuration settings of Travis CI are stored in the ```.travis.yml``` file, including:

	* The language of the project to be built (in my case, it's Ruby for Jekyll), and its version number

			language: ruby

			rvm:
			- 2.0.0

	* The shell script to be executed as the build process on each commit

			script: ./cibuild.sh

	* The branch that Travis CI will listen to for new commits

			branch:
				only:
				- website-template

3. GitHub Token for Travis CI

	Within the build script,  we will need to clone the generated site-content to the ```master``` branch from the Jekyll-built branch. To do so, we can use HTTPS protocol and personal access token for authentication (see [*here*](https://gist.github.com/grawity/4392747) for a comparison among difference methods). The idea of a personal access token is to work as our regular password for the access to GitHub. There is a GitHub article on how to create a personal access token - [*Creating an access token for command-line use*](https://help.github.com/articles/creating-an-access-token-for-command-line-use/).

	By running the following codes in the directory where the generated site content will be cloned (the ```master``` branch in my case), Travis command line tool can encrypt the token and output it as an environment variable in the ```.travis.yml``` file.

		gem install travis
		travis encrypt GH_TOKEN=<the generated token> --add env.global

	As a result, the following should be seen in the ```.travis.yml``` file:

		env:
			global:
			- secure: (encrypted token)

4. Travis CI Build Script

	The main tasks in the build script are:

	* Build the site with Jekyll

	* Clean up or remove old files in the old ```master``` branch

	* Clone ```master``` branch using the encrypted GitHub personal access token for authentication

	* Copy generated site content to ```master``` branch

	* Commit and push the changes

I found [*this link*](http://eshepelyuk.github.io/2014/10/28/automate-github-pages-travisci.html) quiet useful for automating a build of GitHub Page project with Jekyll and Travis CI.

<br>

#### Summary

<br>


Now I've built my first personal blog. It should be quite easy to maintain and update with all the components that I integrated into it.

Overall, this blog is built on GitHub Pages to host the website from my GitHub project repository, Bootstrap as the front-end framework, Disqus to host the blog commenting system, Jekyll as the Static Site Generator and Travis CI for the continuous integration.

Further down the track, this blog will be about my activities and projects in tech.





