from pathlib import Path

import pytest

from agentic_rag.builder.chunker import _extract_title, _parse_sections, chunk_markdown


def _write_markdown(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_title():
    assert _extract_title("# Hello World\n\nSome text") == "Hello World"
    assert _extract_title("No title here") == ""


def test_parse_sections():
    content = "# Title\n\nIntro paragraph.\n\n## Section 1\n\nBody 1.\n\n## Section 2\n\nBody 2."
    sections = _parse_sections(content)
    headings = [heading for heading, _ in sections if heading is not None]
    assert "Title" in headings
    assert "Section 1" in headings
    assert "Section 2" in headings


def test_chunk_markdown_requires_level_one_title(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "## Abstract\n\nThis file does not define a level-one paper title.\n",
    )

    with pytest.raises(ValueError, match="level-1 title heading"):
        chunk_markdown(md)


@pytest.mark.parametrize(
    ("masthead", "paper_title"),
    [
        ("Journal of Materials Chemistry B", "Bacterial S-layer protein inspired multifunctional peptide"),
        ("M A T E R I A L S S C I E N C E", "Contribution of biomimetic collagen-ligand interaction to intrafibrillar mineralization"),
        ("P H Y S I C A L S C I E N C E", "Atomic-scale compositional mapping reveals Mg-rich amorphous calcium phosphate"),
        ("RESEARCH ARTICLE", "Bioactive Glass Empowered Mineralizable Hydrogel for Effective Dentinal Tubule Occlusion"),
        ("ARTICLE OPEN", "Bioengineered alpha-Hairpin peptide for intrafibrillar remineralization"),
    ],
)
def test_chunk_markdown_skips_masthead_h1_and_selects_real_paper_title(
    tmp_path: Path,
    masthead: str,
    paper_title: str,
):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        f"# {masthead}\n\n"
        f"# {paper_title}\n\n"
        "## Abstract\n\n"
        "This abstract paragraph describes the actual article content and should stay with the true paper title.\n\n"
        "## Introduction\n\n"
        "This introduction paragraph should be chunked after the selected paper title and should not include the masthead.\n",
    )

    _, title, _, chunks = chunk_markdown(md)

    assert title == paper_title
    combined = "\n".join(chunk.text for chunk in chunks)
    assert masthead not in combined
    assert paper_title in chunks[0].text


def test_chunk_markdown_cleans_acs_just_accepted_front_matter_and_dedupes_repeated_title(tmp_path: Path):
    paper_title = (
        "Designing solid materials from their solute state: a shift in paradigms towards a holistic approach "
        "in functional materials chemistry"
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Perspective\n\n"
        f"# {paper_title}\n\n"
        "Denis Gebauer, and Stephan E Wolf\n\n"
        "J. Am. Chem. Soc., Just Accepted Manuscript • Publication Date (Web): 12 Feb 2019\n\n"
        "Downloaded from http://pubs.acs.org on February 12, 2019\n\n"
        "# Just Accepted\n\n"
        "“Just Accepted” manuscripts have been peer-reviewed and accepted for publication. They are posted "
        "online prior to technical editing, formatting for publication and author proofing. The American "
        "Chemical Society provides “Just Accepted” as a service to the research community to expedite the "
        "dissemination of scientific material as soon as possible after acceptance.\n\n"
        f"# {paper_title}\n\n"
        "Denis Gebauer,†,‡,* and Stephan E. Wolf§,‡,⁋,*\n\n"
        "†Department of Chemistry, Physical Chemistry, University of Konstanz, Universitätsstraße 10, "
        "78457 Konstanz, Germany\n\n"
        "ABSTRACT: Non-classical notions consider formation pathways of crystalline materials where larger "
        "species than monomeric chemical constituents play crucial roles in nucleation and crystallization.\n\n"
        "# INTRODUCTION\n\n"
        "Across all subfields of chemistry, nucleation and crystallization are phenomena of fundamental "
        "importance and literally ubiquitous.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "ABSTRACT: Non-classical notions consider formation pathways" in combined
    assert "Across all subfields of chemistry" in combined
    assert "Just Accepted” manuscripts have been peer-reviewed" not in combined
    assert "American Chemical Society provides" not in combined
    assert "Downloaded from http://pubs.acs.org" not in combined
    assert "Just Accepted Manuscript" not in combined


def test_chunk_markdown_dedupes_repeated_title_when_later_h1_contains_inline_math_markup(tmp_path: Path):
    plain_title = (
        "Directed Assembly of Cuprous Oxide Nanocatalyst for CO2 Reduction Coupled to "
        "Heterobinuclear ZrOCoII Light Absorber in Mesoporous Silica"
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Article\n\n"
        f"# {plain_title}\n\n"
        "Wooyul Kim, and Heinz Frei\n\n"
        "ACS Catal., Just Accepted Manuscript • DOI: 10.1021/acscatal.5b01306 • Publication Date (Web): 24 Aug 2015\n\n"
        "Downloaded from http://pubs.acs.org on August 25, 2015\n\n"
        "# Just Accepted\n\n"
        "“Just Accepted” manuscripts have been peer-reviewed and accepted for publication. The American "
        "Chemical Society provides “Just Accepted” as a free service to the research community.\n\n"
        "# Directed Assembly of Cuprous Oxide Nanocatalyst for $\\mathbf { C O } _ { 2 }$ Reduction Coupled to "
        "Heterobinuclear ZrOCoII Light Absorber in Mesoporous Silica\n\n"
        "Wooyul Kimand, Heinz Frei*\n\n"
        "Physical Biosciences Division, Lawrence Berkeley National Laboratory, University of California, Berkeley, "
        "CA 94720, United States\n\n"
        "e-mail: HMFrei@lbl.edu\n\n"
        "# ABSTRACT\n\n"
        "This abstract paragraph should remain after the front matter is dropped and the repeated title with "
        "inline math markup is recognized as the same paper title.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == plain_title
    assert "$\\mathbf" not in title
    assert "This abstract paragraph should remain" in combined
    assert "Downloaded from http://pubs.acs.org" not in combined
    assert "Just Accepted” manuscripts have been peer-reviewed" not in combined


def test_chunk_markdown_keeps_body_text_that_mentions_just_accepted_without_heading_trigger(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Mention Test Paper\n\n"
        "# ABSTRACT\n\n"
        "This abstract paragraph discusses the Just Accepted publication workflow as historical context, but it "
        "is real article prose and must not be removed simply because those two words appear in the sentence.\n\n"
        "# INTRODUCTION\n\n"
        "This introduction paragraph should also remain available for chunking and retrieval.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == "Mention Test Paper"
    assert "This abstract paragraph discusses the Just Accepted publication workflow" in combined
    assert "This introduction paragraph should also remain available" in combined


def test_chunk_markdown_accepts_real_all_caps_paper_title(tmp_path: Path):
    paper_title = "PREPARATION AND PROPERTIES OF NOVEL BIOCOMPATIBLEPECTIN/SILICA CALCIUM PHOSPHATE HYBRIDS"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        f"# {paper_title}\n\n"
        "RAGAB E. ABOUZEID, AMAL H. ABD EL-KADER, AHMED SALAMA, TAMER Y. A. FAHMY and MOHAMED EL-SAKHAWY\n\n"
        "Cellulose and Paper Department, National Research Centre, 33 El-Bohouth Str., Dokki, P.O. 12622, "
        "Giza, Egypt\n\n"
        "Corresponding author: M. El-Sakhawy elsakhawy@yahoo.com\n\n"
        "Received January 10, 2022\n\n"
        "The development of bioactive polysaccharide-based hybrid materials is necessary for finding new "
        "alternatives in the field of biomaterials. As a bioactive water-soluble polysaccharide, pectin was "
        "used in this study to prepare reinforced silica gel monoliths through the sol-gel method. In-situ "
        "mineralization of calcium phosphate was achieved using calcium chloride and phosphate precursors. "
        "The properties of the pectin/silica/calcium phosphate hybrid were examined using FTIR, XRD and "
        "SEM/EDX techniques. Based on the results of the tests on kidney (Vero) cell lines, the "
        "pectin/silica/calcium phosphate hybrid demonstrated very mild cytotoxicity.\n\n"
        "Keywords: pectin, silica, calcium phosphate, composite\n\n"
        "# INTRODUCTION\n\n"
        "Bone is a multifunctional organ that is essential to life.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "The development of bioactive polysaccharide-based hybrid materials" in combined
    assert any(chunk.section_hint and chunk.section_hint.lower() == "introduction" for chunk in chunks)


def test_chunk_markdown_skips_author_list_h1_after_real_title(tmp_path: Path):
    paper_title = "Hydrogen generation via ethanol steam reforming over Co/HAp catalysts"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "Contents lists available at ScienceDirect\n\n"
        "# Journal of the Energy Institute\n\n"
        f"# {paper_title}\n\n"
        "# Q1 J. Dobosz* , M. Małecka, M. Zawadzki\n\n"
        "Institute of Low Temperature and Structure Research, Department of Nanomaterials Chemistry and Catalysis, "
        "Polish Academy of Sciences, PO Box 1410, 50-950 Wroclaw, Poland\n\n"
        "# a r t i c l e i n f o\n\n"
        "Article history:\n\n"
        "Received 31 October 2016\n\n"
        "Keywords:\n\n"
        "Hydrogen production\n\n"
        "# a b s t r a c t\n\n"
        "The catalytic activity of calcium hydroxyapatite supported cobalt nanoparticles in ethanol steam reforming "
        "was investigated and the best catalytic properties were obtained over 5%Co/HAp catalyst.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "The catalytic activity of calcium hydroxyapatite supported cobalt nanoparticles" in combined
    assert "Q1 J. Dobosz" not in combined
    assert "Journal of the Energy Institute" not in combined


def test_chunk_markdown_dedupes_taylor_francis_title_variants_and_drops_regular_article_front_matter(tmp_path: Path):
    primary_title = "Ion association in lithium metaborate solution: a Raman and ab initio insight"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        f"# {primary_title}\n\n"
        "Fayan Zhu, Yongquan Zhou, Chunhui Fang, Yan Fang, Haiwen Ge & Hongyan Liu\n\n"
        "To cite this article: Fayan Zhu et al. (2016): Ion association in lithium metaborate solution: "
        "a Raman and ab initio insight, Physics and Chemistry of Liquids, DOI: 10.1080/00319104.2016.1183003\n\n"
        "To link to this article: http://dx.doi.org/10.1080/00319104.2016.1183003\n\n"
        "View supplementary material\n\n"
        "Published online: 21 May 2016.\n\n"
        "Submit your article to this journal\n\n"
        "Article views: 1\n\n"
        "View related articles\n\n"
        "View Crossmark data\n\n"
        "# REGULAR ARTICLE\n\n"
        "# Ion association in lithium metaborate solution: a Raman and insight\n\n"
        "Fayan Zhua, Yongquan Zhoua, Chunhui Fanga, Yan Fanga, Haiwen Gea and Hongyan Liua,b\n\n"
        "a Key Laboratory of Salt Resources and Chemistry, Qinghai Institute of Salt Lakes, Chinese Academy "
        "of Sciences, Xining, Qinghai, China\n\n"
        "# ABSTRACT\n\n"
        "Ion association and hydration clusters in aqueous lithium borate solution are extremely important to "
        "understand some extraordinary properties of lithium borates.\n\n"
        "# ARTICLE HISTORY\n\n"
        "Received 29 January 2016 Accepted 22 April 2016\n\n"
        "# KEYWORDS\n\n"
        "Lithium metaborate; aqueous solution; ion association; DFT\n\n"
        "# 1 Introduction\n\n"
        "Micro-properties of aqueous solutions have attracted researchers' attention through decades.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == primary_title
    assert "Ion association and hydration clusters in aqueous lithium borate solution" in combined
    assert "Micro-properties of aqueous solutions have attracted researchers' attention" in combined
    assert "REGULAR ARTICLE" not in combined
    assert "To cite this article:" not in combined
    assert "View Crossmark data" not in combined
    assert "ARTICLE HISTORY" not in combined
    assert "Lithium metaborate; aqueous solution; ion association; DFT" not in combined


def test_chunk_markdown_treats_later_h1_after_abstract_as_section_heading(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Chemical reactivity under nanoconfinement\n\n"
        "Angela B. Grommet, Moran Feller and Rafal Klajn\n\n"
        "Confining molecules can fundamentally change their chemical and physical properties. Confinement "
        "effects are considered instrumental at various stages of the origins of life, and life continues to "
        "rely on layers of compartmentalization to maintain an out-of-equilibrium state and efficiently "
        "synthesize complex biomolecules under mild conditions. As interest in synthetic confined systems grows, "
        "we are realizing that the principles governing reactivity under confinement are the same in abiological "
        "systems as they are in nature.\n\n"
        "# Acceleration of chemical reactions\n\n"
        "Within a range of different nanospaces, confinement has been shown to accelerate the rate of chemical "
        "reactions, both catalytically and stoichiometrically.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == "Chemical reactivity under nanoconfinement"
    assert "Acceleration of chemical reactions" in combined
    assert any(chunk.section_hint == "Acceleration of chemical reactions" for chunk in chunks)


def test_chunk_markdown_treats_roman_numeral_h1_as_section_heading(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "![](images/cover.jpg)\n\n"
        "Check for updates\n\n"
        "Cite this: Soft Matter, 2018, 14, 7246\n\n"
        "Received 18th January 2018, Accepted 5th August 2018\n\n"
        "DOI: 10.1039/c8sm00133b\n\n"
        "rsc.li/soft-matter-journal\n\n"
        "# DMA study of water’s glass transition in nanoscale confinement\n\n"
        "V. Soprunyuk and W. Schranz *\n\n"
        "Dynamic mechanical analysis measurements of water confined in nanoporous silica have been performed "
        "as a function of temperature and frequency for different pore sizes. Most of the data show three "
        "processes where the characteristic shift with pore size corresponds to freezing or melting of internal "
        "water in the core of the pores. Dynamic elastic measurements show clear signatures of glass freezing "
        "of this supercooled water.\n\n"
        "# I Introduction\n\n"
        "Water is of fundamental importance for all living organisms as well as for abiotic environments.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == "DMA study of water’s glass transition in nanoscale confinement"
    assert "I Introduction" in combined
    assert any(chunk.section_hint == "I Introduction" for chunk in chunks)


def test_chunk_markdown_cleans_accepted_article_front_matter_and_selects_later_real_h1(tmp_path: Path):
    paper_title = "Recent Advance in Interfacial Assembly Growth of Mesoporous Silica on Magnetite Particles"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "A Journal of the Gesellschaft Deutscher Chemiker\n\n"
        "# A Journal of the Gesellschaft Deutscher Chemiker Angewandte GDCh International Edition Chemie\n\n"
        "International Edition\n\n"
        "www.angewandte.org\n\n"
        "# Accepted Article\n\n"
        f"Title: {paper_title}\n\n"
        "Authors: Yonghui Deng, Qin Yue, Jianguo Sun, and Yijin Kang\n\n"
        f"# {paper_title}\n\n"
        "## Abstract\n\n"
        "This abstract describes the interfacial assembly growth of mesoporous silica on magnetite particles "
        "and should remain after accepted-article front matter is removed.\n\n"
        "# Introduction\n\n"
        "The introduction discusses magnetite particles and mesoporous silica growth mechanisms.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "This abstract describes the interfacial assembly growth" in combined
    assert "Accepted Article" not in combined
    assert "Title:" not in combined
    assert "Authors:" not in combined
    assert "Gesellschaft Deutscher Chemiker" not in combined
    assert "www.angewandte.org" not in combined


def test_chunk_markdown_cleans_white_rose_repository_front_matter_and_selects_real_title(tmp_path: Path):
    paper_title = "Effect of Nanoscale Confinement on the Crystallization of Potassium Ferrocyanide"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "Deposited via The University of Leeds.\n\n"
        "White Rose Research Online URL for this paper: https://eprints.whiterose.ac.uk/id/eprint/104873/\n\n"
        "Version: Accepted Version\n\n"
        "# Article:\n\n"
        "Anduix-Claro, C, Kim, YY, Wang, Y et al. (2016) Effect of Nanoscale Confinement on the Crystallization "
        "of Potassium Ferrocyanide. Crystal Growth and Design, 16 (9). pp. 5403-5411.\n\n"
        "https://doi.org/10.1021/acs.cgd.6b00894\n\n"
        "# Reuse\n\n"
        "Items deposited in White Rose Research Online are protected by copyright, with all rights reserved unless "
        "indicated otherwise.\n\n"
        "# Takedown\n\n"
        "If you consider content in White Rose Research Online to be in breach of UK law, please notify us by "
        "emailing eprints@whiterose.ac.uk including the URL of the record and the reason for the withdrawal request.\n\n"
        f"# {paper_title}\n\n"
        "Clara Anduix-Canto, Yi-Yeoun Kim, Yunwei Wang and Hugo K. Christenson\n\n"
        "# ABSTRACT\n\n"
        "Many crystallization processes of great significance in nature and technology occur in small volumes "
        "rather than in bulk solution.\n\n"
        "# INTRODUCTION\n\n"
        "A range of common approaches exist to control crystallization processes in bulk solution.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "Many crystallization processes of great significance" in combined
    assert "A range of common approaches exist to control crystallization processes" in combined
    assert "White Rose Research Online" not in combined
    assert "Items deposited in White Rose Research Online" not in combined
    assert "withdrawal request" not in combined


def test_chunk_markdown_cleans_white_rose_rsc_front_matter_and_skips_journal_masthead(tmp_path: Path):
    paper_title = (
        "Crystallization of citrate-stabilized amorphous calcium phosphate to nanocrystalline apatite: "
        "a surface-mediated transformation"
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "Deposited via The University of York.\n\n"
        "White Rose Research Online URL for this paper: https://eprints.whiterose.ac.uk/id/eprint/98103/\n\n"
        "Version: Accepted Version\n\n"
        "# Article:\n\n"
        "Chatzipanagis, Konstantinos et al. (2016) Crystallization of citrate-stabilized amorphous calcium "
        "phosphate to nanocrystalline apatite: a surface-mediated transformation.\n\n"
        "https://doi.org/10.1039/C6CE00521G\n\n"
        "# Reuse\n\n"
        "Items deposited in White Rose Research Online are protected by copyright.\n\n"
        "# Takedown\n\n"
        "If you consider content in White Rose Research Online to be in breach of UK law, please notify us.\n\n"
        "# CrystEngComm\n\n"
        "Accepted Manuscript\n\n"
        "This article can be cited before page numbers have been issued.\n\n"
        "This is an Accepted Manuscript, which has been through the Royal Society of Chemistry peer review process "
        "and has been accepted for publication.\n\n"
        f"# {paper_title}\n\n"
        "Received 00th January 20xx, Accepted 00th January 20xx\n\n"
        "# ABSTRACT\n\n"
        "This work explores the mechanisms underlying the crystallization of citrate-functionalized amorphous "
        "calcium phosphate in relevant aqueous media.\n\n"
        "# INTRODUCTION\n\n"
        "Many aspects of the mechanisms underlying the formation of nanocrystalline apatite remain under debate.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "This work explores the mechanisms underlying the crystallization" in combined
    assert "Many aspects of the mechanisms underlying the formation of nanocrystalline apatite" in combined
    assert "White Rose Research Online" not in combined
    assert "This article can be cited before page numbers have been issued" not in combined
    assert "Royal Society of Chemistry peer review process" not in combined
    assert "CrystEngComm" not in combined


def test_chunk_markdown_cleans_soft_matter_accepted_manuscript_front_matter(tmp_path: Path):
    paper_title = "A facile route towards PDMAEMA homopolymer amphiphiles"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Soft Matter\n\n"
        "Accepted Manuscript\n\n"
        "This article can be cited before page numbers have been issued, to do this please use: T. Manouras, "
        "Soft Matter, 2017, DOI: 10.1039/C7SM00365J.\n\n"
        "This is an Accepted Manuscript, which has been through the Royal Society of Chemistry peer review "
        "process and has been accepted for publication.\n\n"
        "# ARTICLE\n\n"
        f"# {paper_title}\n\n"
        "Received 00th January 20xx, Accepted 00th January 20xx\n\n"
        "DOI: 10.1039/x0xx00000x\n\n"
        "# ABSTRACT\n\n"
        "Well-defined poly(2-(dimethylamino)ethyl methacrylate) has been modified at low degrees of "
        "quaternization using alkyl halides with long alkyl chains as the quaternization agents.\n\n"
        "# 1 Introduction\n\n"
        "Poly(2-(dimethylamino)ethyl methacrylate) is a well-known dual stimuli-responsive polymer.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "Well-defined poly(2-(dimethylamino)ethyl methacrylate)" in combined
    assert "Poly(2-(dimethylamino)ethyl methacrylate) is a well-known dual stimuli-responsive polymer" in combined
    assert "This is an Accepted Manuscript" not in combined
    assert "can be cited before page numbers have been issued" not in combined
    assert "Soft Matter" not in combined


def test_chunk_markdown_merges_split_title_h1_fragments_in_accepted_front_matter(tmp_path: Path):
    paper_title = (
        "Qualitative Discussion of Prenucleation Cluster Role in Crystallization of Calcium Carbonate "
        "under High Magnesium Concentration"
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Accepted Manuscript\n\n"
        "# Qualitative Discussion of Prenucleation Cluster Role in\n\n"
        "# Crystallization of Calcium Carbonate under High\n\n"
        "# Magnesium Concentration\n\n"
        "Received 00th January 20xx, Accepted 00th January 20xx\n\n"
        "ABSTRACT: This accepted-manuscript example should be indexed under the merged paper title rather than "
        "three separate H1 fragments.\n\n"
        "# INTRODUCTION\n\n"
        "This introduction paragraph should remain available after the split H1 title is merged.\n",
    )

    _, title, _, chunks = chunk_markdown(md)

    assert title == paper_title
    assert chunks[0].text.startswith(paper_title)
    assert "merged paper title rather than three separate H1 fragments" in chunks[0].text
    assert "This introduction paragraph should remain available" in "\n".join(chunk.text for chunk in chunks)


def test_chunk_markdown_uses_last_repeated_title_and_drops_aip_page_noise(tmp_path: Path):
    paper_title = "Assembly of tubes in the stretching-dominated limit"
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "RESEARCH ARTICLE | JULY 21 2025\n\n"
        f"# {paper_title}\n\n"
        "![](images/header.jpg)\n\n"
        "Carlos I. Mendoza ; David Reguera\n\n"
        "Check for updates\n\n"
        "J. Chem. Phys. 163, 034905 (2025)\n\n"
        "https://doi.org/10.1063/5.0271521\n\n"
        "View Online\n\n"
        "Export Citation\n\n"
        "# Articles You May Be Interested In\n\n"
        "On the influence of bending energy on the assembly of spherical viral capsids\n\n"
        "J. Chem. Phys. (August 2025)\n\n"
        "<details>\n\n"
        "<summary>text_image</summary>\n\n"
        "VHFLI Very High Frequency Lock in Amplifier Signal Input Signal Output\n\n"
        "</details>\n\n"
        "The New VHFLI 200 MHz Lock-in Amplifier.\n\n"
        f"# {paper_title}\n\n"
        "Cite as: J. Chem. Phys. 163, 034905 (2025); doi: 10.1063/5.0271521 Submitted: 18 March 2025 "
        "Accepted: 28 June 2025 Published Online: 21 July 2025\n\n"
        "Carlos I. Mendoza1,a) and David Reguera1,2,b)\n\n"
        "# AFFILIATIONS\n\n"
        "1 Departament de Física de la Matèria Condensada, Universitat de Barcelona, Martí i Franquès 1, "
        "08028 Barcelona, Spain\n\n"
        "a)Author to whom correspondence should be addressed: cmendoza@materiales.unam.mx.\n\n"
        "# ABSTRACT\n\n"
        "Many biological and nanotechnological processes rely on the self-assembly of tubular structures. "
        "In this work, we study the conditions under which tubules are self-assembled spontaneously from free "
        "subunits in solution and determine the final radii distribution of the assembled tubes.\n\n"
        "Published under an exclusive license by AIP Publishing. https://doi.org/10.1063/5.0271521\n\n"
        "# I. INTRODUCTION\n\n"
        "Self-assembly is the spontaneous process by which molecules, nanoparticles, and other building blocks "
        "organize themselves into ordered structures without external guidance.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == paper_title
    assert "Many biological and nanotechnological processes rely on the self-assembly of tubular structures" in combined
    assert "Self-assembly is the spontaneous process" in combined
    assert any(chunk.section_hint == "I. INTRODUCTION" for chunk in chunks)
    assert "Articles You May Be Interested In" not in combined
    assert "On the influence of bending energy on the assembly of spherical viral capsids" not in combined
    assert "VHFLI Very High Frequency Lock in Amplifier" not in combined
    assert "The New VHFLI 200 MHz Lock-in Amplifier." not in combined
    assert "View Online" not in combined
    assert "Export Citation" not in combined
    assert "Cite as:" not in combined
    assert "AFFILIATIONS" not in combined
    assert "correspondence should be addressed" not in combined
    assert "Published under an exclusive license by AIP Publishing" not in combined


def test_chunk_markdown_dedupes_repeated_aip_title_with_spacing_or_hyphen_ocr_variation(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "RESEARCH ARTICLE | FEBRUARY 02 2023\n\n"
        "# Deciphering strontium sulfate precipitation via Ostwald’s rule of stages: From prenucleation clusters "
        "to solutionmediated phase tranformation\n\n"
        "Special Collection: Nucleation: Current Understanding Approaching 150 Years After Gibbs\n\n"
        "A. R. Lauer; R. Hellmann ; G. Montes-Hernandez\n\n"
        "Check for updates\n\n"
        "J. Chem. Phys. 158, 054501 (2023)\n\n"
        "https://doi.org/10.1063/5.0136870\n\n"
        "View Online\n\n"
        "Export Citation\n\n"
        "CrossMark\n\n"
        "<details>\n\n"
        "<summary>natural_image</summary>\n\n"
        "Abstract digital artwork with glowing red and blue bokeh effects against a dark blue background\n\n"
        "</details>\n\n"
        "The Journal of Chemical Physics\n\n"
        "2024 Emerging Investigators Special Collection\n\n"
        "Submit Today\n\n"
        "# Deciphering strontium sulfate precipitation via Ostwald’s rule of stages: From prenucleation clusters "
        "to solution-mediated phase tranformation\n\n"
        "Cite as: J. Chem. Phys. 158, 054501 (2023); doi: 10.1063/5.0136870\n\n"
        "Submitted: 29 November 2022 • Accepted: 10 January 2023 • Published Online: 2 February 2023\n\n"
        "A. R. Lauer,1 R. Hellmann,1 G. Montes-Hernandez,1 and A. E. S. Van Driessche1,4,a)\n\n"
        "# AFFILIATIONS\n\n"
        "1 Université Grenoble Alpes, Université Savoie Mont Blanc, CNRS, IRD, ISTerre, 38000 Grenoble, France\n\n"
        "a)Author to whom correspondence should be addressed: alexander.vd@csic.es\n\n"
        "# ABSTRACT\n\n"
        "Multiple-step nucleation pathways have been observed during mineral formation in both inorganic and "
        "biomineral systems. These pathways can involve precursor aqueous species, amorphous intermediates, "
        "or metastable phases.\n",
    )

    _, title, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert title == (
        "Deciphering strontium sulfate precipitation via Ostwald’s rule of stages: From prenucleation clusters "
        "to solution-mediated phase tranformation"
    )
    assert "Multiple-step nucleation pathways have been observed" in combined
    assert "2024 Emerging Investigators Special Collection" not in combined
    assert "Submit Today" not in combined
    assert "CrossMark" not in combined


def test_chunk_markdown_rejects_accepted_article_without_real_h1_title(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "A Journal of the Gesellschaft Deutscher Chemiker\n\n"
        "# A Journal of the Gesellschaft Deutscher Chemiker Angewandte GDCh International Edition Chemie\n\n"
        "# Accepted Article\n\n"
        "Title: Recent Advance in Interfacial Assembly Growth of Mesoporous Silica on Magnetite Particles\n\n"
        "Authors: Yonghui Deng, Qin Yue, Jianguo Sun, and Yijin Kang\n\n"
        "## Abstract\n\n"
        "This file only has a Title metadata field and no real level-one paper title.\n",
    )

    with pytest.raises(ValueError, match="level-1 title heading"):
        chunk_markdown(md)


def test_chunk_markdown_rejects_ambiguous_multiple_paper_title_h1s(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Collagen mineralization in hydrated fibrils under nanoconfinement\n\n"
        "# Calcium phosphate nucleation in hydrated collagen fibrils under confinement\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is only here to make the markdown look like a real paper while keeping the title "
        "selection intentionally ambiguous.\n",
    )

    with pytest.raises(ValueError, match="multiple plausible level-1 title headings"):
        chunk_markdown(md)


def test_chunk_markdown_rejects_multi_abstract_file_with_multiple_plausible_titles(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "Promotion effect of hyaluronic acid on intrafibrillar mineralization of collagen.md",
        "Theme: Oral Health and Systemic Health\n\n"
        "# FC070\n\n"
        "# Developing a validated oral health screening tool for cardiac patients\n\n"
        "Aim or Purpose: This is another conference abstract in the same source file.\n\n"
        "# Theme: Caries\n\n"
        "# FC074\n\n"
        "# Promotion effect of hyaluronic acid on intrafibrillar mineralization of collagen\n\n"
        "Aim or Purpose: This is the abstract that should be selected because it matches the file name.\n\n"
        "# FC075\n\n"
        "# Effect of application time of 38% SDF on ECC (RCT)\n\n"
        "Aim or Purpose: This is yet another conference abstract in the same source file.\n",
    )

    with pytest.raises(ValueError, match="multiple plausible level-1 title headings"):
        chunk_markdown(md)


def test_chunk_markdown_rejects_masthead_only_h1s(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Journal of Materials Chemistry B\n\n"
        "# RESEARCH ARTICLE\n\n"
        "# Abstract\n\n"
        "This file has no real paper title in H1 form and should fail indexing.\n",
    )

    with pytest.raises(ValueError, match="level-1 title heading"):
        chunk_markdown(md)


def test_chunk_markdown_cleans_front_matter_and_references(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Test Paper\n\n"
        "Jane Doe, John Smith\n\n"
        "Department of Biomaterials, Test University\n\n"
        "# A R T I C L E I N F O\n\n"
        "Received 1 January 2025\n\n"
        "# Keywords:\n\n"
        "Biomineralization\n\n"
        "# A B S T R A C T\n\n"
        "This abstract describes early intrafibrillar mineralization in detail and provides enough context "
        "to count as the title-and-abstract chunk for indexing.\n\n"
        "Statement of significance: This summary should stay with the abstract because it explains why the "
        "finding matters for biomimetic mineralization.\n\n"
        "# 1. Introduction\n\n"
        "The introduction explains the biological context for collagen mineralization and provides the first "
        "full body paragraph for the section.\n\n"
        "![](images/fig1.jpg)\n\n"
        "Fig. 1. This caption should be retained because it describes the microscopy evidence that accompanies "
        "the paragraph.\n\n"
        "# References\n\n"
        "1. This reference entry must not be indexed.\n",
    )

    _, title, _, chunks = chunk_markdown(md)

    assert title == "Test Paper"
    assert chunks

    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Jane Doe" not in combined
    assert "Test University" not in combined
    assert "Keywords" not in combined
    assert "This reference entry must not be indexed" not in combined
    assert "Statement of significance" in combined
    assert "Fig. 1." in combined

    first_chunk = chunks[0].text
    assert "Test Paper" in first_chunk
    assert "Abstract" in first_chunk
    assert "This abstract describes early intrafibrillar mineralization" in first_chunk
    assert "Statement of significance" in first_chunk


def test_chunk_markdown_infers_abstract_and_preserves_pre_heading_body(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Inferred Abstract Paper\n\n"
        "Alice Smith, Bob Jones, Carol White\n\n"
        "School of Materials Science, Example University\n\n"
        "This first substantial paragraph acts as the abstract because it appears before the first formal "
        "section heading and summarizes the key observation about collagen mineralization with enough detail "
        "to qualify as a real abstract paragraph in the cleaned corpus.\n\n"
        "This second paragraph should be treated as introductory body text even though the source file does "
        "not provide an explicit introduction heading before the first formal subsection begins.\n\n"
        "# Results\n\n"
        "This results paragraph should remain available as a later section.\n",
    )

    _, _, _, chunks = chunk_markdown(md)

    assert chunks[0].section_hint == "Abstract"
    assert "This first substantial paragraph acts as the abstract" in chunks[0].text
    intro_chunk = next(chunk for chunk in chunks if chunk.section_hint == "Introduction")
    assert "This second paragraph should be treated as introductory body text" in intro_chunk.text


def test_chunk_markdown_drops_long_author_lists_and_affiliations(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Long Author List Paper\n\n"
        "Li-na Niu1†, Sang Eun Jee2†, Kai Jiao1, Lige Tonggu3, Mo Li3, Liguo Wang3, Yao-dong Yang4, "
        "Ji-hong Bian4, Lorenzo Breschi5, Seung Soon Jang2, Ji-hua Chen1, David H. Pashley6 and Franklin R. Tay6\n\n"
        "Department of Biomaterials, Max Planck Institute of Colloids and Interfaces, Potsdam 14476, Germany\n\n"
        "This first real sentence-like paragraph should be inferred as the abstract because it is the first "
        "substantial prose paragraph before the formal section headings begin in the document, and it continues "
        "with enough extra detail about collagen mineralization, precursor infiltration, and early crystal growth "
        "to satisfy the substantial-paragraph threshold used by the chunker.\n\n"
        "## Introduction\n\n"
        "This introduction paragraph should remain.\n\n"
        "Prof. Z. Zou, Dr. M. Alberic, Prof. Y. Politi, Dr. L. Bertinetti\n\n"
        "E-mail: test@example.com\n\n"
        "This second introduction paragraph should also remain after the noisy author footer is removed.\n",
    )

    _, _, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Li-na Niu1" not in combined
    assert "Department of Biomaterials" not in combined
    assert "Prof. Z. Zou" not in combined
    assert "test@example.com" not in combined
    assert "This first real sentence-like paragraph should be inferred as the abstract" in combined
    assert "This second introduction paragraph should also remain" in combined


def test_chunk_markdown_drops_intro_pollution_from_author_footers(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Polluted Introduction Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is clean and should remain in the first chunk because it is real article prose.\n\n"
        "## 1. Introduction\n\n"
        "This real introduction paragraph should remain before the noisy footer material.\n\n"
        "Potsdam 14476, Germany\n\n"
        "Prof. Z. Zou, Q. Wang\n\n"
        "The ORCID identification number(s) for the author(s) of this article can be found under https://doi.org/10.1002/example.\n\n"
        "DOI: 10.1002/example\n\n"
        "© 2020 The Authors. Published by WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim. This is an open access article.\n\n"
        "This later introduction paragraph should still remain after those noisy lines are removed.\n",
    )

    _, _, _, chunks = chunk_markdown(md)
    intro_text = "\n".join(chunk.text for chunk in chunks if chunk.section_hint == "1. Introduction")
    assert "This real introduction paragraph should remain" in intro_text
    assert "This later introduction paragraph should still remain" in intro_text
    assert "Potsdam 14476, Germany" not in intro_text
    assert "Prof. Z. Zou" not in intro_text
    assert "ORCID identification number" not in intro_text
    assert "DOI: 10.1002/example" not in intro_text
    assert "Published by WILEY-VCH" not in intro_text


def test_chunk_markdown_drops_long_concatenated_affiliations(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Affiliation Block Paper\n\n"
        "Author One, Author Two, Author Three\n\n"
        "a Development Department, ITOCHU Mineral Resources Development Corp., 5-1, Kita-Aoyama 2-chome, Minato-ku, Tokyo, Japan "
        "b Graduate School of Engineering, Kyoto University, Katsura C1-2-215, Kyoto 615-8540, Japan "
        "c Geothermal Department, West Japan Engineering Consultants Inc., 1-1, 1-chome, Watanabe-dori, Chuo-Ku, Fukuoka, Japan\n\n"
        "# A B S T R A C T\n\n"
        "This abstract paragraph should remain after the long affiliation block is removed from the cleaned output.\n\n"
        "# 1. Introduction\n\n"
        "This introduction paragraph should remain as the first body chunk.\n",
    )

    _, _, _, chunks = chunk_markdown(md)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Development Department" not in combined
    assert "Graduate School of Engineering" not in combined
    assert "Geothermal Department" not in combined
    assert "This abstract paragraph should remain" in combined


def test_short_first_paragraph_merges_with_second_paragraph(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Short Section Paper\n\n"
        "## Abstract\n\n"
        "A sufficiently long abstract paragraph that makes the first chunk independent from the body and keeps "
        "this test focused on the section-level grouping logic in the chunker implementation.\n\n"
        "## Results\n\n"
        "Very short lead paragraph.\n\n"
        "This second paragraph is much longer and should be merged with the short lead paragraph instead of "
        "forming a separate tiny chunk that would cut the local semantics apart.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=1200, chunk_overlap=0)
    results_chunk = next(chunk for chunk in chunks if chunk.section_hint == "Results")
    assert "Very short lead paragraph." in results_chunk.text
    assert "This second paragraph is much longer" in results_chunk.text


def test_caption_merges_with_adjacent_body_chunk(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Caption Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is only here to keep the structure of the markdown file close to a real paper.\n\n"
        "## Introduction\n\n"
        "This body paragraph discusses the experiment immediately before the figure.\n\n"
        "![](images/fig1.jpg)\n\n"
        "Fig. 1. Caption text describing the figure should stay with nearby evidence instead of becoming its own "
        "standalone chunk.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=1200, chunk_overlap=0)
    intro_chunks = [chunk for chunk in chunks if chunk.section_hint == "Introduction"]
    assert any(
        "This body paragraph discusses the experiment immediately before the figure." in chunk.text
        and "Fig. 1. Caption text describing the figure" in chunk.text
        for chunk in intro_chunks
    )


def test_short_sections_pack_across_headings(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Section Packing Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is only here to keep the title-and-abstract chunk separate from the body "
        "section packing regression case.\n\n"
        "## 2.3. Mineralization\n\n"
        "Fig. 3. Mined veins and main faults on the 0M level overlain on the topographic map of the "
        "Hosokura deposit area around the Odogamori dacite lava dome.\n\n"
        "## 3. Reconstruction of temperature and fluid flow settings by mineralization stage\n\n"
        "The ore deposition mechanism during early, middle, and late stages was refined using mineralogical "
        "and fluid inclusion analyses.\n\n"
        "## 3.1. Paleo-temperature distribution\n\n"
        "There are two effective methods for estimating the paleo-temperatures of hydrothermal fluids in veins "
        "and adjacent host rocks, and these methods were used for each mineralization stage.\n\n"
        "## 3.1.1. Early stage\n\n"
        "A total of 108 samples of hydrothermally altered rocks from outcrops in the Hosokura deposit area and "
        "underground tunnels around the Fuji vein were collected and used for powder X-ray diffraction analyses. "
        "The alteration halo around the vein was classified into six zones based on the appearance and "
        "disappearance of adularia and clay minerals.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=1200, chunk_overlap=0)
    mineralization_chunk = next(chunk for chunk in chunks if "2.3. Mineralization" in chunk.text)

    assert mineralization_chunk.section_hint == "2.3. Mineralization"
    assert "Fig. 3. Mined veins and main faults" in mineralization_chunk.text
    assert "3. Reconstruction of temperature and fluid flow settings by mineralization stage" in mineralization_chunk.text
    assert "The ore deposition mechanism during early, middle, and late stages" in mineralization_chunk.text


def test_chunk_markdown_preserves_hierarchical_heading_path_in_text_and_section_hint(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Hierarchy Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph keeps the title-and-abstract chunk separate from the hierarchical body regression.\n\n"
        "## 2. Experimental section\n\n"
        "### 2.1. The preparation of lignin/chitosan composites beads derived N-doped highly microporous carbon\n\n"
        "#### 2.1.1. The synthesis of lignin/chitosan composites beads by the self-assembly method\n\n"
        "Due to the inter-molecular forces between lignin and chitosan, including hydrogen bond, electrostatic "
        "action, and van der Waals force, which provide a good chemical basis for the self-assembly of the "
        "two natural polymers to form composite beads and stable adsorption precursors for carbonization.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=1200, chunk_overlap=0)
    hierarchy_chunk = next(
        chunk
        for chunk in chunks
        if chunk.section_hint
        == "2. Experimental section > 2.1. The preparation of lignin/chitosan composites beads derived N-doped highly microporous carbon > 2.1.1. The synthesis of lignin/chitosan composites beads by the self-assembly method"
    )

    assert "2. Experimental section" in hierarchy_chunk.text
    assert "2.1. The preparation of lignin/chitosan composites beads derived N-doped highly microporous carbon" in hierarchy_chunk.text
    assert "2.1.1. The synthesis of lignin/chitosan composites beads by the self-assembly method" in hierarchy_chunk.text
    assert "Due to the inter-molecular forces between lignin and chitosan" in hierarchy_chunk.text


def test_long_hierarchical_section_repeats_full_heading_path_across_chunks(tmp_path: Path):
    long_paragraph = " ".join(
        f"Sentence {index} explains a detailed adsorption observation for lignin chitosan composite beads."
        for index in range(1, 61)
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Hierarchy Split Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph keeps the split regression focused on the hierarchical body section behavior.\n\n"
        "## 2. Experimental section\n\n"
        "### 2.1. Preparation of samples\n\n"
        "#### 2.1.1. Synthesis of composites beads\n\n"
        f"{long_paragraph}\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=700, chunk_overlap=80)
    section_chunks = [
        chunk
        for chunk in chunks
        if chunk.section_hint == "2. Experimental section > 2.1. Preparation of samples > 2.1.1. Synthesis of composites beads"
    ]

    assert len(section_chunks) >= 2
    assert all("2. Experimental section" in chunk.text for chunk in section_chunks)
    assert all("2.1. Preparation of samples" in chunk.text for chunk in section_chunks)
    assert all("2.1.1. Synthesis of composites beads" in chunk.text for chunk in section_chunks)


def test_adjacent_hierarchical_subsections_repeat_shared_parent_path_when_packed(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Hierarchy Packing Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph keeps the title-and-abstract chunk separate from the subsection packing regression.\n\n"
        "## 2. Experimental section\n\n"
        "### 2.1. Preparation of samples\n\n"
        "#### 2.1.1. First subsection\n\n"
        "This short paragraph describes the first subsection and is intentionally brief.\n\n"
        "#### 2.1.2. Second subsection\n\n"
        "This second short paragraph should be packed into the same chunk while repeating the shared parent headings.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=1200, chunk_overlap=0)
    packed_chunk = next(chunk for chunk in chunks if "2.1.1. First subsection" in chunk.text)

    assert "2.1.2. Second subsection" in packed_chunk.text
    assert packed_chunk.text.count("2. Experimental section") >= 2
    assert packed_chunk.text.count("2.1. Preparation of samples") >= 2


def test_section_heading_stays_with_following_paragraph_after_boundary_flush(tmp_path: Path):
    results_paragraph = " ".join(
        f"Sentence {index} describes a detailed result about mineral formation."
        for index in range(1, 12)
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Heading Boundary Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is only here to keep the structure realistic before the body chunks begin.\n\n"
        "## Results\n\n"
        f"{results_paragraph}\n\n"
        "## Discussion\n\n"
        "This discussion paragraph must stay with the Discussion heading instead of leaving the heading as a "
        "standalone tail at the previous chunk boundary.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=700, chunk_overlap=0)
    discussion_chunk = next(chunk for chunk in chunks if "Discussion" in chunk.text)

    assert "This discussion paragraph must stay with the Discussion heading" in discussion_chunk.text
    assert not any(chunk.text.rstrip().endswith("Discussion") for chunk in chunks)


def test_long_paragraph_splits_but_keeps_section_heading(tmp_path: Path):
    long_paragraph = " ".join(
        f"Sentence {index} explains a detailed observation about mineral infiltration and crystal growth."
        for index in range(1, 61)
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Long Paragraph Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph is long enough to stand on its own but short enough that it should not split.\n\n"
        "## Results\n\n"
        f"{long_paragraph}\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=700, chunk_overlap=80)
    result_chunks = [chunk for chunk in chunks if chunk.section_hint == "Results"]
    assert len(result_chunks) >= 2
    assert all("Results" in chunk.text for chunk in result_chunks)


def test_table_body_is_dropped_but_caption_is_kept(tmp_path: Path):
    table_rows = "".join(
        f"<tr><td>AD{index}</td><td>{index * 1.23:.2f}</td><td>{index * 4.56:.2f}</td></tr>"
        for index in range(1, 80)
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Table Cleaning Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph keeps the paper structure valid before testing body table cleaning.\n\n"
        "## Results\n\n"
        "Table 1. Experimental adsorption values used for model fitting.\n\n"
        f"<table><tr><th>Code</th><th>Value A</th><th>Value B</th></tr>{table_rows}</table>\n\n"
        "| Code | Value A | Value B |\n\n"
        "| --- | ---: | ---: |\n\n"
        "| AD80 | 12.3 | 45.6 |\n\n"
        "The text after the table should remain available for retrieval.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=700, chunk_overlap=0)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert "Table 1. Experimental adsorption values used for model fitting." in combined
    assert "The text after the table should remain available for retrieval." in combined
    assert "<table>" not in combined
    assert "<td>" not in combined
    assert "| AD80 |" not in combined


def test_flowchart_markup_is_dropped_but_figure_caption_is_kept(tmp_path: Path):
    mermaid = " ".join(
        f'A{index}["Dilute dehydrogenation barrier"] --> A{index + 1}["Dilute dehydrogenation barrier"]'
        for index in range(1, 120)
    )
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Flowchart Cleaning Paper\n\n"
        "## Abstract\n\n"
        "This abstract paragraph keeps the paper structure valid before testing flowchart cleaning.\n\n"
        "## Results\n\n"
        f"A total of <details> <summary>flowchart</summary> ```mermaid graph TD {mermaid} ``` </details> Figure 1.\n\n"
        "Figure 1. Schematic illustration of the remineralization procedure for demineralized dentin.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=700, chunk_overlap=0)
    combined = "\n".join(chunk.text for chunk in chunks)

    assert "Figure 1. Schematic illustration of the remineralization procedure" in combined
    assert "<details>" not in combined
    assert "```mermaid" not in combined
    assert "Dilute dehydrogenation barrier" not in combined


def test_chunk_markdown_repairs_broken_sentence_paragraphs(tmp_path: Path):
    md = _write_markdown(
        tmp_path,
        "paper.md",
        "# Broken Paragraph Paper\n\n"
        "## 2.1. Deposit geology\n\n"
        "K–Ar dating of adularia from one vein indicated a Late Miocene age of 5.8 ± 0.2 Ma, "
        "consistent with the age of the dacite lava dome generated during the caldera-formation period. "
        "The veins are thought to have been formed in tensile fractures caused by the uplift of the deposit area "
        "during caldera formation. The Akakura caldera, formed during the\n\n"
        "Pliocene to Pleistocene, is situated approximately 20-km southwest of the Hosokura deposit and hosts "
        "a fossil geothermal system.\n",
    )

    _, _, _, chunks = chunk_markdown(md, chunk_size=600, chunk_overlap=0)
    geology_chunks = [chunk for chunk in chunks if chunk.section_hint == "2.1. Deposit geology"]
    combined = "\n".join(chunk.text for chunk in geology_chunks)

    assert "The Akakura caldera, formed during the Pliocene to Pleistocene" in combined
    assert not any(
        chunk.text.rstrip().endswith("formed during the")
        for chunk in geology_chunks
    )
