

#____________________________________________________________________________________________________________________________________

def begin() -> str:

    return """\n\n\n"""


# ____________________________________________________________________________________________________________________________________


def add_chapter(chapter_title: str) -> str:
    return r"""

\clearpage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%
%%%

\chapter{*section_name}

%%%
%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Discussion}

""".replace('*section_name', chapter_title)


#____________________________________________________________________________________________________________________________________


def add_section(section_title: str) -> str:

    return r"""
    
\clearpage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%
%%%

\section{*section_name}

%%%
%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Discussion}

""".replace('*section_name', section_title)


# ____________________________________________________________________________________________________________________________________


def add_subsection(subsection_title: str) -> str:
    return r"""

\clearpage

%%%%%%%%%%%%%%
%%%%%%%%
%%%

\subsection{*section_name}

%%%
%%%%%%%%
%%%%%%%%%%%%%%

\subsection{Discussion}

""".replace('*section_name', subsection_title)

#____________________________________________________________________________________________________________________________________


def add_figure(fig_path: str,
               fig_caption: str,
               fig_label: str,
               width_times_textwidth: float = 1,
               ) -> str:

    return r"""

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{figure}[!h]
\centering
\includegraphics[width=*width\textwidth]{*fig_path}
\caption{*fig_caption}
\label{*fig_label}
\end{figure}

""".replace('*fig_path', fig_path).replace('*fig_caption', fig_caption).replace('*fig_label', fig_label).replace('*width', str(width_times_textwidth))

#____________________________________________________________________________________________________________________________________


# def add_table()

























