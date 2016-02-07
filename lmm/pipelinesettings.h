#ifndef PIPELINESETTINGS_H
#define PIPELINESETTINGS_H

#include <QVariant>

class QSettings;

class PipelineSettings
{
public:
	static PipelineSettings * getInstance();

	void setBackendFile(const QString &filename);
	virtual QVariant get(const QString &setting);
	virtual int set(const QString &setting, const QVariant &value);
	virtual bool isEqual(const QString &setting, const QString &val);

protected:
	explicit PipelineSettings();

	QSettings *sets;

	static PipelineSettings *inst;
};

#endif // PIPELINESETTINGS_H
