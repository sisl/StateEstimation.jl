html"""<style>
pluto-notebook {
  counter-reset: section;
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
"""