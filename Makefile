SCRIPTS_REPO := "https://github.com/mycard/ygopro-scripts.git"
SCRIPTS_DIR := "../ygopro-scripts"
DATABASE_REPO := "https://github.com/mycard/ygopro-database/raw/7b1874301fc1aa52bd60585589f771e372ff52cc/locales"
LOCALES := en zh

.PHONY: all assets script py_install ygoenv_so clean dev

all: assets script py_install

dev: assets script py_install ygoenv_so

py_install:
	pip install -e ygoenv
	pip install -e ygoinf
	pip install -e .

ygoenv_so: ygoenv/ygoenv/ygopro/ygopro_ygoenv.so

ygoenv/ygoenv/ygopro/ygopro_ygoenv.so:
	xmake b ygopro_ygoenv

script : scripts/script 

scripts/script:
	if [ ! -d $(SCRIPTS_DIR) ] ; then git clone $(SCRIPTS_REPO) $(SCRIPTS_DIR); fi
	cd $(SCRIPTS_DIR) && git checkout 8e7fde9
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

clean:
	rm -rf scripts/script
	rm -rf assets/locale/en assets/locale/zh