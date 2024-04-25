SCRIPTS_REPO := "https://github.com/mycard/ygopro-scripts.git"
SCRIPTS_DIR := "../ygopro-scripts"
DATABASE_REPO := "https://github.com/mycard/ygopro-database/raw/master/locales"
LOCALES := en zh

.PHONY: all assets script py_install ygoenv_so

all: assets script py_install

py_install: ygoenv_so
	pip install -e ygoenv
	pip install -e .

ygoenv_so: ygoenv/ygoenv/ygopro/ygopro_ygoenv.so

ygoenv/ygoenv/ygopro/ygopro_ygoenv.so:
	xmake b ygopro_ygoenv

script : scripts/script 

scripts/script:
	if [ ! -d $(SCRIPTS_DIR) ] ; then git clone $(SCRIPTS_REPO) $(SCRIPTS_DIR); fi
	ln -sf "../$(SCRIPTS_DIR)" scripts/script

assets: $(LOCALES)

$(LOCALES): % : assets/locale/%/cards.cdb assets/locale/%/strings.conf

assets/locale/en assets/locale/zh:
	mkdir -p $@

assets/locale/en/cards.cdb: assets/locale/en
	wget -nv $(DATABASE_REPO)/en-US/cards.cdb -O $@

assets/locale/en/strings.conf: assets/locale/en
	wget -nv $(DATABASE_REPO)/en-US/strings.conf -O $@

assets/locale/zh/cards.cdb: assets/locale/zh
	wget -nv $(DATABASE_REPO)/zh-CN/cards.cdb -O $@

assets/locale/zh/strings.conf: assets/locale/zh
	wget -nv $(DATABASE_REPO)/zh-CN/strings.conf -O $@