import bs4
import lxml
from lxml import etree

def parse_cei(charter_path):
    with open(f"{charter_path}/cei.xml", "r", encoding="utf-8") as file:
        tree = etree.parse(file)
        abstract = "".join(get_xpath_result(f"{mapping['cei:abstract']}/descendant-or-self::text()[not(self::cei:sup)]"))
        return {"abstract": abstract}
        # atom_id.append("".join(get_xpath_result(f"{mapping['atom:id']}/text()")))
        # cei_abstract.append("".join(get_xpath_result(f"{mapping['cei:abstract']}/descendant-or-self::text()[not(self::cei:sup)]")))
        # cei_abstract_foreign.append(get_xpath_result(f"{mapping['cei:foreign']}/text()"))
        # cei_tenor.append("".join(get_xpath_result(f"{mapping['cei:tenor']}/descendant-or-self::text()[not(self::cei:sup)]")))
        # cei_placeName.append(get_xpath_result(f"{mapping['cei:issued/cei:placeName']}/text()"))
        # cei_lang_MOM.append(get_xpath_result(f"{mapping['cei:lang_MOM']}/text()"))
        # cei_date.append(get_xpath_result(f"{mapping['cei:date']}/text()"))
        # cei_dateRange.append(get_xpath_result(f"{mapping['cei:dateRange']}/text()"))
        # cei_date_ATTRIBUTE_value.append(get_xpath_result(f"{mapping['cei:date']}/@value"))
        # cei_dateRange_ATTRIBUTE_from.append(get_xpath_result(f"{mapping['cei:dateRange']}/@from"))
        # cei_dateRange_ATTRIBUTE_to.append(get_xpath_result(f"{mapping['cei:dateRange']}/@to"))
        # cei_graphic_ATTRIBUTE_url_orig.append(get_xpath_result(f"{mapping['@url']}"))
        # cei_graphic_ATTRIBUTE_url_copy.append(get_xpath_result(f"{mapping['cei:graphic/@url']}"))

def extract_abstracts(charter_path, mode="text"):
    assert mode in ("text",)
    namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'cei': 'http://www.monasterium.net/NS/cei'}
    with open(f"{charter_path}/cei.xml", "r", encoding="utf-8") as file:
        tree = etree.parse(file)
        abstract = tree.xpath('/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:abstract', namespaces=namespaces)

        #abstract = "".join(get_xpath_result(f"{mapping['cei:abstract']}/    descendant-or-self::text()[not(self::cei:sup)]"))
        return {"abstract": abstract}