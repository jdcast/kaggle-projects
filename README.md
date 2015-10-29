# kaggle-projects
Repository for holding kaggle projects
=============

Requirements
============

* python 2.7
* anaconda

Installation
============
* `git clone git@github.com:jdcast/kaggle-projects.git`
* `cd kaggle-projects`

Todo
====

git workflow
============

* work on a branch cut from master: `git co -b features/the-name-of-your-story`
* rebase your branch before merging: `git rebase master`
* merge with `--no-ff` to create merge commits: `git merge features/the-name-of-your-story --no-ff`
* commit early and often
* commit messages: present tense and begin with a capital: 'Add cookies to jar'
* when your branch is ready, `git push` and create a pull request for it on github

Tools (following is only for linux/os-x: vim + tmux + git)
=====

* (expansive unix command cheat sheet here: https://ubuntudanmark.dk/filer/fwunixref.pdf, be careful with chmod :P)
* vundle (plugin manager): https://github.com/gmarik/Vundle.vim 
* example of using vundle: https://github.com/kchmck/vim-coffee-script#install-using-vundle
* support for vim + tmux: http://fideloper.com/mac-vim-tmux (follow all steps) 
* jdcast's .vimrc, .gitconfig, .etc: https://github.com/jdcast/config-files (these will take care of remaining items below if you choose to use them out-of-the-box)
* add .gitignore to ~/ directory (if not already there) with following at top so as to not create/push swap files to repo: 
```
*.swp
*.swo
```
* add .gitconfig to ~/ directory (if not already there) with following at top so as to enable automatic reference management when using `git push` so that `git push <branch>` translates to simply `git push`:
```
[push]
	default = current
```
* add .vimrc to ~/ directory (if not already there).  Folliwng is a good starting point for .vimrc (this will accomplish all of the required steps for the file in the above walkthroughs but feel free to expand and share :) ):
```
set nocompatible
filetype off    " Required

set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" Required
Bundle 'VundleVim/Vundle.vim'   

" solarized theme
Bundle 'altercation/vim-colors-solarized'

" coffeescript
Bundle 'kchmck/vim-coffee-script'

" jade
Bundle 'digitaltoad/vim-jade'

" JSON
Bundle 'elzr/vim-json'

call vundle#end()

" Don't create swap files
set noswapfile

" Some settings to enable the theme:

" Show line numbers
set number 

" Show matching brackets
set showmatch

" Do case insensitive matching
set ignorecase

set ruler
set nowrap

set tabstop=2
set shiftwidth=2
set expandtab

"set completion-ignore-case on
"set show-all-if-abmiguous on
"TAB: menu-complete

" Use syntax highlighting
syntax enable

set background=dark

let g:solarized_termcolors = 256

colorscheme solarized


" Required
filetype plugin indent on
``` 
