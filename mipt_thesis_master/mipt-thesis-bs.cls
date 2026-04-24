\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mipt-thesis}[2014/06/18 Class for the BS thesis in the MIPT]

%%%%%% Class options %%%%%
\DeclareOption*{
    \ClassWarning{myclass}{Unknown option '\CurrentOption'}
}
\ProcessOptions\relax

\LoadClass[a4paper,12pt]{book}
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Useful packages %%%%%
% Math packages, should be loaded before fontenc and babel for Cyrillic
\RequirePackage{amsmath,amsthm,amssymb}
\RequirePackage{mathtext}
% Fonts and lang
\RequirePackage[T1,TS1,T2A]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english,russian]{babel}
% Formatting
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{nopageno}
\RequirePackage{tocloft}
% Style file with additional packages
\RequirePackage{mipt-thesis}
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% TL 2026+ compatibility: один из пакетов выше (babel 26.x / geometry 6.0 /
%%%%% siunitx 3.4.x) оставляет открытую группу, из-за чего ядро LaTeX 2025-11
%%%%% ругается "Loading a class or package in a group" на каждом \usepackage
%%%%% в main.tex и (каскадно) l3sockets падает внутри hyperref. Закрываем
%%%%% «лишние» группы перед тем, как определять собственные команды класса.
%%%%% NB: \loop здесь не сработает (его \body локален и теряется после первого
%%%%% \endgroup); используем рекурсивный \gdef, который переживает \endgroup.
%%%%% (Внутри .cls символ @ уже является буквой, \makeatletter не нужен.)
\gdef\mt@flush@groups{\ifnum\currentgrouplevel>0 \endgroup \mt@flush@groups \fi}
\mt@flush@groups
%%%%% TL 2026 fix: убрали \begingroup\globaldefs=1\relax. Этот шаблон в TL
%%%%% 2026 ломает внутренние LaTeX3 хелперы команд-парсера: при \renewcommand
%%%%% под globaldefs=1 \@notdefinable/\@ifdefinable «протекают» в глобальный
%%%%% хеш и потом hyperref/etoolbox падают с ложными «already defined» на
%%%%% \undef и \footnoteautorefname. Определения и так находятся на верхнем
%%%%% уровне .cls (group level 0) — им нечего «переживать», поэтому групповая
%%%%% обёртка не нужна. \mt@flush@groups выше гарантирует GL=0.

%%%%% Set up correct formatting %%%%%
\geometry{left=3cm}
\geometry{right=2cm}
\geometry{top=2cm}
\geometry{bottom=2cm}
% NB: \onehalfspace — это ENVIRONMENT из setspace (открывает \begingroup),
% а не команда установки межстрочного интервала. Раньше код «работал»
% только потому, что \begingroup\globaldefs=1 ниже превращал все определения
% в глобальные, и утечка группы была безвредной. После удаления globaldefs
% утечка ломает классовские \newcommand'ы (они становятся локальными в
% группе, открытой \onehalfspace, и теряются на выходе из неё). Используем
% \onehalfspacing — преамбульная команда без группы.
\onehalfspacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Some dark TeX magic %%%%%
\addto\captionsrussian{\renewcommand{\contentsname}{Содержание}}
\addto\captionsrussian{\renewcommand{\listfigurename}{Список иллюстраций}}
\addto\captionsrussian{\renewcommand{\listtablename}{Список таблиц}}
\addto\captionsrussian{\renewcommand{\bibname}{Литература}}
\addto\captionsrussian{\renewcommand{\indexname}{Index}}
\addto\captionsrussian{\renewcommand{\figurename}{Рис.{}}}
\addto\captionsrussian{\renewcommand{\tablename}{Таблица}}
\addto\captionsrussian{\renewcommand{\partname}{Часть}}
\addto\captionsrussian{\renewcommand{\chaptername}{Глава}}
\addto\captionsrussian{\renewcommand{\appendixname}{Приложение}}
%%% Has no effect since babel package switches tags (see https://tex.stackexchange.com/questions/153182/strange-behaviour-on-redefining-contentsname, /usr/share/texlive/texmf-dist/tex/generic/babel-russian/russianb.ldf (Debian))%%%
%\renewcommand\contentsname{Содержание}
%\renewcommand\listfigurename{Список иллюстраций}
%\renewcommand\listtablename{Список таблиц}
%\renewcommand\bibname{Литература}
%\renewcommand\indexname{Index}
%\renewcommand\figurename{Рис.{}}
%\renewcommand\tablename{Таблица}
%\renewcommand\partname{Часть}
%\renewcommand\chaptername{Глава}
%\renewcommand\appendixname{Приложение}


\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \partmark{#1}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries\boldmath \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries\boldmath #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries\boldmath #1\par}%
    \@endpart}

\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#2}%
                       \else
                         \addcontentsline{toc}{chapter}{#2}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#2}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}


\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries\boldmath \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries\boldmath #1\par\nobreak
    \vskip 40\p@
  }}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries\boldmath  #1\par\nobreak
    \vskip 40\p@
  }}


\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                             {\normalfont\Large\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                             {\normalfont\Large\sffamily\itshape\raggedright}}
%%%%%%%% Captions %%%%%%%%%%%%%%%%%%%%%%%%%%
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1. #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1. #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%%%%%%%% Headings
\fancypagestyle{miptthesis}{%
    \fancyhf{}
    \fancyhead[RO, LE]{\thepage}
    \fancyhead[CO]{\leftmark}
    \fancyhead[CE]{\rightmark}
}
\pagestyle{miptthesis}
\renewcommand{\chaptermark}[1]{\markboth{\hfill\normalfont #1\hfill}{}}
\renewcommand{\sectionmark}[1]{\markright{\hfill\large\itshape \S\thesection\quad #1\hfill}}
\let\partmark=\chaptermark

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Right TOC numbering %%%%%
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftsecaftersnum}{.}
\renewcommand{\cftsubsecaftersnum}{.}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Easier references %%%%%
\newcommand{\formref}[1]{(\ref{#1})}
\newcommand{\picref}[1]{рис.~\ref{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Titlepage and contents %%%%%
% Set some "variables"
\renewcommand{\author}[1]{\def\mtauthortext{#1}}
\renewcommand{\title}[1]{\def\mttitletext{#1}}
\newcommand{\groupnum}[1]{\def\mtgroupnumtext{#1}}
\newcommand{\supervisor}[1]{\def\mtsupervisortext{#1}}
\newcommand{\faculty}[1]{\def\mtfacultytext{#1}}
\newcommand{\department}[1]{\def\mtdepartmenttext{#1}}

\newcommand{\addemptypage}{%
    \clearpage \mbox{} \clearpage
}

\renewcommand{\titlepage}{%
    \pagestyle{empty}
    \begin{center}
        {\bf Министерство образования и науки Российской Федерации \\
        Московский физико-технический институт (национальный 
        исследовательский университет) \\ }
        \vspace{1cm}


        \mtfacultytext \\
        \mtdepartmenttext \\

        \vspace{3em}

        Выпускная квалификационная работа магистра по направлению 09.04.01
        <<Информатика и вычислительная техника >>
    \end{center}

    \begin{center}
        \vspace{\fill}
        \singlespacing

        \LARGE \mttitletext{}

        \vspace{\fill}
    \end{center}


    \begin{flushright}
        Студент \mtgroupnumtext{} группы \\
        \mtauthortext{} \\
        \vspace{2em}
        Научный руководитель \\
        \mtsupervisortext{} \\
    \end{flushright}

    \vspace{7em}

    \begin{center}
        Долгопрудный \\
        \the\year{}
    \end{center}

}

\newcommand{\titlecontents}{
    \titlepage \addemptypage
    \newpage
    \tableofcontents \addemptypage
    \pagestyle{miptthesis}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% (нет \endgroup — \begingroup\globaldefs=1 выше удалён в рамках TL 2026 fix)

%%%%% На всякий случай ещё раз проверяем группу: если собственные определения
%%%%% класса открыли группу, закрываем её — чтобы main.tex начинал при GL=0.
\mt@flush@groups

\endinput
