---
layout: post_entry
title: The Development of This Blog (II)
---

Following the previous article on "The development of this blog (I)", the second part of the article will discuss the following components:

* [Static Site Generator - Jekyll](#static-site-generator)

* [Continous Integration - Travis](#continous-integration)


<br>

#### <a name="static-site-generator"></a>Static Site Generator - Jekyll

<br>

At this point, I have a website that includes HTML files for the structure and content, and CSS files for the style and layout. However, each time I want to update an existed post or add a new one, I will have to edit the HTML files or create a new one and copy the structure as well as the layout across respectively, which is not efficient at all. Ideally, I would like to separate the content from the structure and style of the web page, so I will only need to focus on the content when I try to update the blog.

This is why static site generator is useful. It is a software that takes text and templates as input and generates HTML files as output to be displayed on the web browsers. The text is the content that we would like to write in our blog, while the templates are the structure, layout and style etc. that don't require much changes over the updates of the web pages. 

[*Here*](https://www.staticgen.com/) shows the top open-source static site generator and status of their popularity among users. I chose Jekyll to generate the static web pages for my blog because it is on the top of the list and has a very large user base.

**Jekyll**

Jekyll is [*a simple blog-aware static generator*](http://jekyllrb.com/docs/home/) written in Ruby. Once Jekyll is installed, it can build the website project and output files into a specific destination folder (*"_site"* by default). Jekyll also comes with a built-in develoment server, where you can preview what the generated site looks like locally.

A basic Jekyll site usually has the following folders and files in the project folder:

+ _config.yml - the file that stores configuration settings, including where things are, handling reading, filtering content, plugins, conversion, serving, outputting, and markdown processors etc.

+ _includes - the folder that contains partials that can be reused by layout templates and posts, for example, the footer or the Disqus commenting system.

+ _layouts - the folder that includes the layout templates of posts. The templates are selected by posts with a [*YAML Front Matter*](http://jekyllrb.com/docs/frontmatter/) at the start of a file.

+ _posts - this is the folder where the content of the website written a selected markup language (I used *Markdown* in this blog) sits. The file-names must follow this format: **YEAR-MONTH-DAY-title.MARKUP** while words are separated by hyphens. The [*permalinks*](http://jekyllrb.com/docs/permalinks/) can be customised in the file _config.yml or each post, but the date and markup language are determined by the file name. At the start of each post file, a YAML Front Matter should be included to indicate which layout template this post is wrapped by. Here is an example of the YAML Front Matter:

		---  
		layout: post_entry  
		title: The Development of This Blog (II)  
		---  

+ index.html - the default file of the web page. With a YAML Front Matter, it can be transformed by Jekyll. Indices of blog posts can be displayed in this file using [*Liquid template language*](https://docs.shopify.com/themes/liquid-documentation/basics). 

+ _site - this folder is where the generated site will be after ```jekyll build``` putting all the layouts and content together. The files in this folder are those needed to display the website in a browser.

Once the files and folders are in place, we could execute Jekyll to generate the site and deploy the files in folder _site to GitHub or other web hosting service:

1. Generate files for the website in the project folder:

	```
	~$ jekyll build
	```

2. Start a development server locally to see what the site will look like:

	```
	~$ jekyll serve
	```

3. Deploy files in folder *_site* to *username.github.io* once the updates are finished. 

**Markdown**

According to the [*Markdown project page*](https://daringfireball.net/projects/markdown/):

>"Markdown is a text-to-HTML conversion tool for web writers. Markdown allows you to write using an easy-to-read, easy-to-write plain text format, then convert it to structurally valid XHTML (or HTML)."

>"Thus, “Markdown” is two things: (1) a plain text formatting syntax; and (2) a software tool, written in Perl, that converts the plain text formatting to HTML."

Users are able to read the text-content easily through the plain text formatting [*syntax*](https://daringfireball.net/projects/markdown/syntax), rather than trying to find the actual content between a group of HTML tags. Then the software tool can convert the plain text format to structureally valid HTML for displaying the web page.

If you use the text editor [*Sublime*](http://www.sublimetext.com/) to edit a Markdown (.md) file, there is also [*Monokai Extended*](https://github.com/jonschlinkert/sublime-monokai-extended), which has additional syntax highlighting for easier editing.

<br>

#### <a name="continous-integration"></a>Continous Integration - Travis

<br>