#include "pipelinesettings.h"
#include "debug.h"

#include <QSettings>

PipelineSettings * PipelineSettings::inst = NULL;

PipelineSettings *PipelineSettings::getInstance()
{
	if (!inst)
		inst = new PipelineSettings;
	return inst;
}

void PipelineSettings::setBackendFile(const QString &filename)
{
	sets = new QSettings(filename, QSettings::IniFormat);
}

QVariant PipelineSettings::get(const QString &setting)
{
	return sets->value(setting);
}

int PipelineSettings::set(const QString &setting, const QVariant &value)
{
	sets->setValue(setting, value);
	sets->sync();
	return 0;
}

bool PipelineSettings::isEqual(const QString &setting, const QString &val)
{
	return get(setting) == val ? true : false;
}

PipelineSettings::PipelineSettings()
{
	sets = new QSettings("/tmp/pipeline.conf", QSettings::IniFormat);
}
