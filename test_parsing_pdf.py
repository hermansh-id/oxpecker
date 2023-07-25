from oxpecker import LayoutExtractor

lp = LayoutExtractor()
data = lp.extract("test/pdf/test_paper.pdf")
print(data)