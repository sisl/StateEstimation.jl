html"""<style>
pluto-notebook {
  counter-reset: section footnote footnote-title;
}
pluto-notebook h2 {
  counter-reset: subsection;
}
pluto-notebook h2:before {
  counter-increment: section;
  content: "" counter(section) ". ";
}
pluto-notebook h3:before {
  counter-increment: subsection;
  content: counter(section) "." counter(subsection) " ";
}

a.footnote {
    font-weight: normal !important;
    font-size: 0 !important;
    vertical-align: super;
}
a.footnote::before {
    counter-increment: footnote;
    content: "" counter(footnote) "";
    font-size: 10px;
}

.footnote-title {
    font-size: 0 !important;
}
.footnote-title::before {
    counter-increment: footnote-title !important;
    content: "[" counter(footnote-title) "]" !important;
    font-size: 0.75rem !important;
}
"""