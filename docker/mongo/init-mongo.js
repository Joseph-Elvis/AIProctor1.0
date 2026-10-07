const dbName = process.env.MONGO_DB || 'aiproctor';
const appUser = process.env.MONGO_APP_USER || 'aiproctor';
const appPassword = process.env.MONGO_APP_PASSWORD || 'change_me';

const db = db.getSiblingDB(dbName);

const existingUser = db.getUser(appUser);
if (!existingUser) {
  db.createUser({
    user: appUser,
    pwd: appPassword,
    roles: [{ role: 'readWrite', db: dbName }]
  });
}
